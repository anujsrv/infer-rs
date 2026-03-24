use candle_core::{DType, Device, Result as CandleResult, Tensor};

use crate::config::ModelConfig;

pub struct LmModel {
    pub config: ModelConfig,
    embedding:  Tensor,   // [vocab_size, hidden_size]
    proj:       Tensor,   // [hidden_size, vocab_size]
    bias:       Tensor,   // [vocab_size]
}

impl LmModel {
    pub fn new(config: ModelConfig) -> CandleResult<Self> {
        let device = Device::Cpu;
        let v = config.vocab_size;
        let d = config.hidden_size;

        let embedding = Tensor::randn(0f32, 0.02f32, (v, d), &device)?;
        let proj      = Tensor::randn(0f32, 0.02f32, (d, v), &device)?;
        let bias      = Tensor::zeros((v,), DType::F32, &device)?;

        Ok(Self { config, embedding, proj, bias })
    }

    // Returns `(next_token_id, logits_vec)`.
    pub fn forward(&self, token_ids: &[usize]) -> CandleResult<(usize, Vec<f32>)> {
        assert!(!token_ids.is_empty(), "token_ids must not be empty");
        let last_id = *token_ids.last().unwrap();

        let emb = self.embedding.get(last_id)?;
        let emb = emb.unsqueeze(0)?;

        let out = emb.matmul(&self.proj)?;
        let out = out.broadcast_add(&self.bias)?;
        let logits = out.squeeze(0)?;

        // probs = exp(x - max(x)) / sum(exp(x - max(x)))
        let max_val = logits.max(0)?;
        let shifted = logits.broadcast_sub(&max_val)?;
        let exp     = shifted.exp()?;
        let sum     = exp.sum(0)?;
        let probs   = exp.broadcast_div(&sum)?;

        let next_id = probs.argmax(0)?.to_scalar::<u32>()? as usize;

        let logits_vec: Vec<f32> = logits.to_vec1()?;
        Ok((next_id, logits_vec))
    }
}
