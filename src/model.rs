pub mod qwen2;

use std::rc::Rc;

use candle_core::{DType, Device, Result, Tensor, D};

// Linear projection
/// `y = x @ W.T (+ b)`, with `W` stored in PyTorch nn.Linear convention
/// (shape `[out_features, in_features]`).
pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul(&self.w.t()?)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None    => Ok(y),
        }
    }
}


// RMSNorm
/// `rmsnorm(x) = x / sqrt(mean(x²) + eps) * w`, normalising along the last dim.
/// The reduction is done in f32 for stability even when input is bf16/fp16.
pub struct RmsNorm {
    pub w:   Tensor,    // [hidden] per-dim gain
    pub eps: f64,
}

impl RmsNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x32      = x.to_dtype(DType::F32)?;
        let var      = x32.sqr()?.mean_keepdim(D::Minus1)?;
        let denom    = (var + self.eps)?.sqrt()?;
        let normed   = x32.broadcast_div(&denom)?.to_dtype(in_dtype)?;
        normed.broadcast_mul(&self.w)
    }
}


// Rotary positional embeddings (RoPE)
/// Precomputed cos/sin tables for RoPE. Llama-/Qwen-style pairing:
/// dims `(i, i + head_dim/2)` form one 2D vector and get rotated together.
pub struct Rotary {
    pub cos: Tensor,    // [max_pos, head_dim/2]
    pub sin: Tensor,    // [max_pos, head_dim/2]
}

impl Rotary {
    /// Build the cos/sin tables for all positions in `0..max_pos`.
    /// `theta` is the RoPE base — 10K for Llama1/GPT, 1M for Qwen2.
    pub fn new(
        head_dim: usize,
        max_pos:  usize,
        theta:    f64,
        device:   &Device,
        dtype:    DType,
    ) -> Result<Self> {
        let half = head_dim / 2;

        // inv_freq[k] = 1 / theta^(2k / head_dim) for k in 0..half
        let inv_freq: Vec<f32> = (0..half)
            .map(|k| (1.0 / theta.powf(2.0 * k as f64 / head_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;

        // positions: [max_pos, 1]
        let positions = Tensor::arange(0u32, max_pos as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;

        // angles[p, k] = p * inv_freq[k]  -> [max_pos, half]
        let angles = positions.broadcast_mul(&inv_freq)?;
        let cos    = angles.cos()?.to_dtype(dtype)?;
        let sin    = angles.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    /// Rotate `q` and `k` by their absolute positions.
    /// Shapes: `[seq, n_heads, head_dim]`.
    /// `start_pos` is the position of the first token in this batch
    /// `v` is intentionally not rotated; the dot product `q · k` is what
    /// consumes positional info.
    pub fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let (seq, _, head_dim) = q.dims3()?;
        let half = head_dim / 2;

        // Slice cos/sin to the positions we need, broadcast across head axis.
        let cos = self.cos.narrow(0, start_pos, seq)?.unsqueeze(1)?;  // [seq, 1, half]
        let sin = self.sin.narrow(0, start_pos, seq)?.unsqueeze(1)?;

        Ok((rotate(q, &cos, &sin, half)?, rotate(k, &cos, &sin, half)?))
    }
}

/// Apply RoPE to a tensor of shape `[seq, n_heads, head_dim]`.
/// Pair convention: dim `i` and dim `i + half` form one 2D vector.
fn rotate(t: &Tensor, cos: &Tensor, sin: &Tensor, half: usize) -> Result<Tensor> {
    let t1 = t.narrow(2, 0,    half)?;    // first  half: [seq, n_heads, half]
    let t2 = t.narrow(2, half, half)?;    // second half: [seq, n_heads, half]
    // [r1, r2] = [[cos, -sin], [sin, cos]] @ [t1, t2]
    let r1 = t1.broadcast_mul(cos)?.broadcast_sub(&t2.broadcast_mul(sin)?)?;
    let r2 = t1.broadcast_mul(sin)?.broadcast_add(&t2.broadcast_mul(cos)?)?;
    Tensor::cat(&[&r1, &r2], 2)
}


// Attention helpers
/// Causal (lower-triangular) attention mask of shape `[n, n]`:
/// `0` on/below the diagonal, `-inf` above. Added to scores before softmax
/// so token `i` can't attend to tokens `> i`.
fn causal_mask(n: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mut data = vec![0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            data[i * n + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (n, n), device)?.to_dtype(dtype)
}

/// Numerically stable softmax along the last dim:
/// subtract the max before exp to avoid overflow.
fn softmax_last_dim(t: &Tensor) -> Result<Tensor> {
    let max     = t.max_keepdim(D::Minus1)?;
    let shifted = t.broadcast_sub(&max)?;
    let exp     = shifted.exp()?;
    let sum     = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&sum)
}

/// Repeat each KV head `n_rep` times to match Q's head count.
/// `[seq, n_kv, head_dim]` -> `[seq, n_kv * n_rep, head_dim]`.
/// Done via reshape/expand so no extra data is copied.
fn repeat_kv(t: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(t.clone());
    }
    let (seq, n_kv, head_dim) = t.dims3()?;
    t.unsqueeze(2)?
     .expand((seq, n_kv, n_rep, head_dim))?
     .contiguous()?
     .reshape((seq, n_kv * n_rep, head_dim))
}


// Attention (GQA + RoPE + causal mask)
pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub rotary:     Rc<Rotary>,
    pub n_heads:    usize,
    pub n_kv_heads: usize,
    pub head_dim:   usize,
}

impl Attention {
    /// Full-sequence forward
    /// Input shape: `[seq, hidden]`; output shape: `[seq, hidden]`.
    /// Steps:
    ///   1. Project to Q, K, V
    ///   2. Reshape into per-head tensors
    ///   3. Apply RoPE to Q and K (positions `0..seq`)
    ///   4. Repeat KV heads to match Q
    ///   5. Transpose to head-major for batched matmul
    ///   6. Scaled dot product: `Q @ K.T / sqrt(head_dim)`
    ///   7. Add causal mask
    ///   8. Softmax along last dim
    ///   9. Weighted sum of V
    ///  10. Transpose back, reshape, output projection
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq, _hidden) = x.dims2()?;
        let head_dim = self.head_dim;
        let n_heads  = self.n_heads;
        let n_kv     = self.n_kv_heads;
        let n_rep    = n_heads / n_kv;

        // 1. QKV projections
        let q = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;

        // 2. Reshape into per-head tensors
        let q = q.reshape((seq, n_heads, head_dim))?;
        let k = k.reshape((seq, n_kv,    head_dim))?;
        let v = v.reshape((seq, n_kv,    head_dim))?;

        // 3. RoPE on Q and K
        let (q, k) = self.rotary.apply(&q, &k, 0)?;

        // 4. Repeat KV heads to match Q
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // 5. Head-major transpose for batched matmul: [n_heads, seq, head_dim]
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;

        // 6. Scaled dot product
        let scale  = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?.contiguous()?)?;
        let scores = (scores * scale)?;

        // 7. Causal mask (broadcasts over the heads axis)
        let mask   = causal_mask(seq, x.device(), x.dtype())?;
        let scores = scores.broadcast_add(&mask)?;

        // 8. Softmax along the keys dim
        let weights = softmax_last_dim(&scores)?;

        // 9. Weighted sum of V: [n_heads, seq, head_dim]
        let attn = weights.matmul(&v)?;

        // 10. Back to [seq, hidden] and project
        let attn = attn.transpose(0, 1)?
                       .contiguous()?
                       .reshape((seq, n_heads * head_dim))?;
        self.o.forward(&attn)
    }
}


// MLP (SwiGLU)
/// SwiGLU FFN: `down( silu(gate(x)) * up(x) )`.
/// `gate` and `up` project hidden -> intermediate; `down` projects back.
pub struct Mlp {
    pub gate: Linear,
    pub up:   Linear,
    pub down: Linear,
}

impl Mlp {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate      = self.gate.forward(x)?;
        let up        = self.up.forward(x)?;
        let activated = silu(&gate)?.mul(&up)?;
        self.down.forward(&activated)
    }
}

/// SiLU / Swish: `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
fn silu(t: &Tensor) -> Result<Tensor> {
    let one = Tensor::ones_like(t)?;
    let sig = (t.neg()?.exp()? + &one)?.recip()?;
    t.mul(&sig)
}


// Block (one transformer layer)
/// One transformer block: pre-norm + attention (+ residual) + pre-norm + MLP (+ residual).
/// `post_norm` is misleadingly named (HF transformers convention) — it's the
/// pre-norm for the MLP.
pub struct Block {
    pub input_norm: RmsNorm,
    pub attn:       Attention,
    pub post_norm:  RmsNorm,
    pub mlp:        Mlp,
}

impl Block {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Residual #1: x + attn(rmsnorm(x))
        let attn_in  = self.input_norm.forward(x)?;
        let attn_out = self.attn.forward(&attn_in)?;
        let h        = (x + attn_out)?;

        // Residual #2: h + mlp(rmsnorm(h))
        let mlp_in  = self.post_norm.forward(&h)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        &h + mlp_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device { Device::Cpu }

    #[test]
    fn linear_forward_shape() {
        let dev = cpu();
        let w = Tensor::randn(0f32, 1f32, (8, 4), &dev).unwrap();
        let b = Tensor::zeros((8,), DType::F32, &dev).unwrap();
        let lin = Linear { w, b: Some(b) };

        let x = Tensor::randn(0f32, 1f32, (3, 4), &dev).unwrap();
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 8]);
    }

    #[test]
    fn linear_no_bias_ones() {
        // W = ones((2,3)), x = ones((1,3)) -> each output dim = 1+1+1 = 3.
        let dev = cpu();
        let w = Tensor::ones((2, 3), DType::F32, &dev).unwrap();
        let lin = Linear { w, b: None };
        let x = Tensor::ones((1, 3), DType::F32, &dev).unwrap();
        let y = lin.forward(&x).unwrap();
        let row: Vec<f32> = y.get(0).unwrap().to_vec1().unwrap();
        assert_eq!(row, vec![3.0, 3.0]);
    }

    #[test]
    fn rmsnorm_ones_input_returns_weight() {
        // For x = ones((1, d)), mean(x²) = 1, denom = sqrt(1+eps) ≈ 1,
        // so rmsnorm(x) ≈ x * w = w.
        let dev = cpu();
        let w_vals = vec![2.0f32, 3.0, 5.0, 7.0];
        let w = Tensor::from_vec(w_vals.clone(), (4,), &dev).unwrap();
        let rn = RmsNorm { w, eps: 1e-6 };
        let x = Tensor::ones((1, 4), DType::F32, &dev).unwrap();
        let y = rn.forward(&x).unwrap();
        let row: Vec<f32> = y.get(0).unwrap().to_vec1().unwrap();
        for (got, want) in row.iter().zip(w_vals.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn rotary_position_zero_is_identity() {
        // At position 0 all angles are 0 -> cos=1, sin=0 -> RoPE = identity.
        let dev = cpu();
        let head_dim = 8;
        let rot = Rotary::new(head_dim, 16, 10_000.0, &dev, DType::F32).unwrap();

        let cos0: Vec<f32> = rot.cos.get(0).unwrap().to_vec1().unwrap();
        let sin0: Vec<f32> = rot.sin.get(0).unwrap().to_vec1().unwrap();
        for c in cos0 { assert!((c - 1.0).abs() < 1e-6); }
        for s in sin0 { assert!(s.abs() < 1e-6); }

        let q = Tensor::randn(0f32, 1f32, (1, 1, head_dim), &dev).unwrap();
        let (qr, _) = rot.apply(&q, &q, 0).unwrap();
        let q_flat: Vec<f32>  = q .flatten_all().unwrap().to_vec1().unwrap();
        let qr_flat: Vec<f32> = qr.flatten_all().unwrap().to_vec1().unwrap();
        for (got, want) in qr_flat.iter().zip(q_flat.iter()) {
            assert!((got - want).abs() < 1e-5, "rotated != original at pos 0");
        }
    }

    #[test]
    fn causal_mask_shape_and_values() {
        let dev = cpu();
        let m = causal_mask(4, &dev, DType::F32).unwrap();
        assert_eq!(m.dims(), &[4, 4]);
        for i in 0..4 {
            let row: Vec<f32> = m.get(i).unwrap().to_vec1().unwrap();
            for j in 0..4 {
                if j <= i {
                    assert_eq!(row[j], 0.0, "mask[{i},{j}] should be 0");
                } else {
                    assert!(row[j].is_infinite() && row[j] < 0.0,
                            "mask[{i},{j}] should be -inf, got {}", row[j]);
                }
            }
        }
    }

    #[test]
    fn softmax_last_dim_rows_sum_to_one() {
        let dev = cpu();
        let x = Tensor::randn(0f32, 1f32, (3, 5), &dev).unwrap();
        let y = softmax_last_dim(&x).unwrap();
        let sums: Vec<f32> = y.sum(D::Minus1).unwrap().to_vec1().unwrap();
        for s in sums {
            assert!((s - 1.0).abs() < 1e-5, "row sum = {s}, want 1.0");
        }
    }

    #[test]
    fn softmax_handles_neg_inf_mask() {
        // -inf should soft-mask the position to exactly 0 probability.
        let dev = cpu();
        let row = vec![1.0f32, 2.0, f32::NEG_INFINITY, 3.0];
        let x = Tensor::from_vec(row, (1, 4), &dev).unwrap();
        let y = softmax_last_dim(&x).unwrap();
        let probs: Vec<f32> = y.get(0).unwrap().to_vec1().unwrap();
        assert_eq!(probs[2], 0.0, "masked position should get probability 0");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn repeat_kv_expands_each_head() {
        let dev = cpu();
        // [seq=1, n_kv=2, head_dim=3]: head 0 = [1,2,3], head 1 = [4,5,6].
        let data = vec![1.0f32, 2.0, 3.0,  4.0, 5.0, 6.0];
        let t = Tensor::from_vec(data, (1, 2, 3), &dev).unwrap();
        let out = repeat_kv(&t, 3).unwrap();
        assert_eq!(out.dims(), &[1, 6, 3]);
        let rows: Vec<Vec<f32>> = (0..6)
            .map(|h| out.get(0).unwrap().get(h).unwrap().to_vec1().unwrap())
            .collect();
        for h in 0..3 {
            assert_eq!(rows[h], vec![1.0, 2.0, 3.0], "head {h} should clone KV head 0");
        }
        for h in 3..6 {
            assert_eq!(rows[h], vec![4.0, 5.0, 6.0], "head {h} should clone KV head 1");
        }
    }

    #[test]
    fn repeat_kv_noop_when_n_rep_is_one() {
        let dev = cpu();
        let t = Tensor::randn(0f32, 1f32, (2, 4, 3), &dev).unwrap();
        let out = repeat_kv(&t, 1).unwrap();
        assert_eq!(out.dims(), &[2, 4, 3]);
    }

    #[test]
    fn silu_known_values() {
        let dev = cpu();
        let x = Tensor::from_vec(vec![0f32, 1.0, -1.0, 5.0], (4,), &dev).unwrap();
        let y = silu(&x).unwrap();
        let v: Vec<f32> = y.to_vec1().unwrap();
        // silu(0) = 0; silu(1) ≈ 0.7311; silu(-1) ≈ -0.2689; silu(5) ≈ 4.9665
        assert!(v[0].abs() < 1e-6);
        assert!((v[1] - 0.7311).abs() < 1e-3);
        assert!((v[2] - (-0.2689)).abs() < 1e-3);
        assert!((v[3] - 4.9665).abs() < 1e-3);
    }

    #[test]
    fn mlp_forward_shape() {
        let dev = cpu();
        let hidden = 4;
        let intermediate = 6;
        let mk_lin = |out: usize, in_: usize| -> Linear {
            Linear {
                w: Tensor::randn(0f32, 0.1f32, (out, in_), &dev).unwrap(),
                b: None,
            }
        };
        let mlp = Mlp {
            gate: mk_lin(intermediate, hidden),
            up:   mk_lin(intermediate, hidden),
            down: mk_lin(hidden, intermediate),
        };
        let x = Tensor::randn(0f32, 1f32, (3, hidden), &dev).unwrap();
        let y = mlp.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, hidden]);
    }

    #[test]
    fn block_forward_shape() {
        // Smoke test: end-to-end block forward keeps the [seq, hidden] shape.
        let dev = cpu();
        let hidden       = 8;
        let n_heads      = 4;
        let n_kv_heads   = 2;
        let head_dim     = hidden / n_heads;
        let intermediate = 16;

        let mk_lin = |out: usize, in_: usize| -> Linear {
            Linear {
                w: Tensor::randn(0f32, 0.1f32, (out, in_), &dev).unwrap(),
                b: None,
            }
        };
        let mk_norm = || -> RmsNorm {
            RmsNorm {
                w:   Tensor::ones((hidden,), DType::F32, &dev).unwrap(),
                eps: 1e-6,
            }
        };

        let block = Block {
            input_norm: mk_norm(),
            attn: Attention {
                q: mk_lin(n_heads    * head_dim, hidden),
                k: mk_lin(n_kv_heads * head_dim, hidden),
                v: mk_lin(n_kv_heads * head_dim, hidden),
                o: mk_lin(hidden, n_heads * head_dim),
                rotary: Rc::new(Rotary::new(head_dim, 16, 10_000.0, &dev, DType::F32).unwrap()),
                n_heads,
                n_kv_heads,
                head_dim,
            },
            post_norm: mk_norm(),
            mlp: Mlp {
                gate: mk_lin(intermediate, hidden),
                up:   mk_lin(intermediate, hidden),
                down: mk_lin(hidden, intermediate),
            },
        };

        let x = Tensor::randn(0f32, 1f32, (5, hidden), &dev).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[5, hidden]);
    }

    #[test]
    fn attention_forward_shape() {
        // Tiny attention with random weights; check output shape only.
        let dev = cpu();
        let hidden     = 8;
        let n_heads    = 4;
        let n_kv_heads = 2;
        let head_dim   = hidden / n_heads;   // 2

        let mk_lin = |out: usize, in_: usize| -> Linear {
            Linear {
                w: Tensor::randn(0f32, 0.1f32, (out, in_), &dev).unwrap(),
                b: None,
            }
        };

        let attn = Attention {
            q: mk_lin(n_heads    * head_dim, hidden),
            k: mk_lin(n_kv_heads * head_dim, hidden),
            v: mk_lin(n_kv_heads * head_dim, hidden),
            o: mk_lin(hidden, n_heads * head_dim),
            rotary: Rc::new(Rotary::new(head_dim, 16, 10_000.0, &dev, DType::F32).unwrap()),
            n_heads,
            n_kv_heads,
            head_dim,
        };

        let x = Tensor::randn(0f32, 1f32, (5, hidden), &dev).unwrap();
        let y = attn.forward(&x).unwrap();
        assert_eq!(y.dims(), &[5, hidden]);
    }

    #[test]
    fn rotary_position_one_known_values() {
        // head_dim = 4 -> half = 2. inv_freq[0] = 1, so the first pair rotates
        // by angle = 1 radian at position 1.
        // For q = [1, 0, 0, 0]: q'[0] = cos(1), q'[2] = sin(1) (since pair = (0,2)).
        let dev = cpu();
        let head_dim = 4;
        let rot = Rotary::new(head_dim, 8, 10_000.0, &dev, DType::F32).unwrap();

        let q_data = vec![
            1.0f32, 0.0, 0.0, 0.0,    // pos 0
            1.0,    0.0, 0.0, 0.0,    // pos 1
        ];
        let q = Tensor::from_vec(q_data, (2, 1, head_dim), &dev).unwrap();
        let (qr, _) = rot.apply(&q, &q, 0).unwrap();
        let pos1: Vec<f32> = qr.get(1).unwrap().get(0).unwrap().to_vec1().unwrap();

        let c = 1f32.cos();
        let s = 1f32.sin();
        assert!((pos1[0] - c).abs() < 1e-5, "dim 0: got {}, want cos(1) = {c}", pos1[0]);
        assert!((pos1[2] - s).abs() < 1e-5, "dim 2: got {}, want sin(1) = {s}", pos1[2]);
        // Pair (1, 3) starts at zero, stays zero.
        assert!(pos1[1].abs() < 1e-5);
        assert!(pos1[3].abs() < 1e-5);
    }
}
