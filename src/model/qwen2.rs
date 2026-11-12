use std::path::Path;
use std::rc::Rc;

use candle_core::{DType, Device, Result, Tensor};

use crate::config::ModelConfig;

use super::{Attention, Block, Linear, Mlp, RmsNorm, Rotary};

/// Full Qwen2-style decoder-only transformer.
///
/// `embed` is reused as the tied LM head (Qwen2.5-0.5B has
/// `tie_word_embeddings = true`).
pub struct Qwen2Model {
    pub embed:      Tensor,     // [vocab, hidden]
    pub blocks:     Vec<Block>,
    pub final_norm: RmsNorm,
    pub config:     ModelConfig,
    pub device:     Device,
    pub dtype:      DType,
}

impl Qwen2Model {
    /// Load weights from `model.safetensors` inside `model_dir`.
    /// All tensors are cast to `dtype` (typically `DType::F32` for v1).
    pub fn load(
        model_dir: &Path,
        device:    &Device,
        dtype:     DType,
        cfg:       &ModelConfig,
    ) -> Result<Self> {
        let raw = candle_core::safetensors::load(
            model_dir.join("model.safetensors"),
            device,
        )?;

        let get = |name: &str| -> Result<Tensor> {
            raw.get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))?
                .to_dtype(dtype)
        };

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary = Rc::new(Rotary::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            device,
            dtype,
        )?);

        let embed = get("model.embed_tokens.weight")?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            blocks.push(Block {
                input_norm: RmsNorm {
                    w:   get(&format!("{p}.input_layernorm.weight"))?,
                    eps: cfg.rms_norm_eps,
                },
                attn: Attention {
                    q: Linear {
                        w: get(&format!("{p}.self_attn.q_proj.weight"))?,
                        b: Some(get(&format!("{p}.self_attn.q_proj.bias"))?),
                    },
                    k: Linear {
                        w: get(&format!("{p}.self_attn.k_proj.weight"))?,
                        b: Some(get(&format!("{p}.self_attn.k_proj.bias"))?),
                    },
                    v: Linear {
                        w: get(&format!("{p}.self_attn.v_proj.weight"))?,
                        b: Some(get(&format!("{p}.self_attn.v_proj.bias"))?),
                    },
                    o: Linear {
                        w: get(&format!("{p}.self_attn.o_proj.weight"))?,
                        b: None,
                    },
                    rotary:     Rc::clone(&rotary),
                    n_heads:    cfg.num_attention_heads,
                    n_kv_heads: cfg.num_key_value_heads,
                    head_dim,
                },
                post_norm: RmsNorm {
                    w:   get(&format!("{p}.post_attention_layernorm.weight"))?,
                    eps: cfg.rms_norm_eps,
                },
                mlp: Mlp {
                    gate: Linear { w: get(&format!("{p}.mlp.gate_proj.weight"))?, b: None },
                    up:   Linear { w: get(&format!("{p}.mlp.up_proj.weight"))?,   b: None },
                    down: Linear { w: get(&format!("{p}.mlp.down_proj.weight"))?, b: None },
                },
            });
        }

        let final_norm = RmsNorm {
            w:   get("model.norm.weight")?,
            eps: cfg.rms_norm_eps,
        };

        Ok(Self {
            embed,
            blocks,
            final_norm,
            config: cfg.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Full forward pass over a token sequence.
    /// Returns logits of shape `[seq, vocab]`. No KV cache — every call
    /// re-processes the entire prefix.
    pub fn forward(&self, tokens: &[u32]) -> Result<Tensor> {
        let n = tokens.len();

        // 1. Embedding lookup: [n] -> [n, hidden]
        let ids   = Tensor::from_vec(tokens.to_vec(), (n,), &self.device)?;
        let mut h = self.embed.index_select(&ids, 0)?.to_dtype(self.dtype)?;

        // 2. 24 transformer blocks
        for block in &self.blocks {
            h = block.forward(&h)?;
        }

        // 3. Final RMSNorm
        let h = self.final_norm.forward(&h)?;

        // 4. Tied LM head: [n, hidden] @ [hidden, vocab] -> [n, vocab]
        h.matmul(&self.embed.t()?.contiguous()?)
    }
}
