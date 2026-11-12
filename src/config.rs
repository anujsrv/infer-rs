use std::path::Path;
use serde::Deserialize;


#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    Llama,
    Mistral,
    Gemma,
    Phi,
    Qwen,
    Unknown(String),
}

impl ModelType {
    fn detect(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama"                       => Self::Llama,
            "mistral"                     => Self::Mistral,
            "gemma" | "gemma2" | "gemma3" => Self::Gemma,
            "phi"   | "phi3"              => Self::Phi,
            "qwen2" | "qwen"              => Self::Qwen,
            other                         => Self::Unknown(other.to_string()),
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama      => write!(f, "Llama"),
            Self::Mistral    => write!(f, "Mistral"),
            Self::Gemma      => write!(f, "Gemma"),
            Self::Phi        => write!(f, "Phi"),
            Self::Qwen       => write!(f, "Qwen"),
            Self::Unknown(s) => write!(f, "Unknown({s})"),
        }
    }
}


// Mirrors every field we may care about in `config.json`.
// All are `Option` so missing fields don't cause parse errors.
#[derive(Debug, Deserialize)]
struct RawHfConfig {
    model_type:              Option<String>,
    architectures:           Option<Vec<String>>,
    vocab_size:              Option<usize>,

    hidden_size:             Option<usize>,
    intermediate_size:       Option<usize>,
    num_hidden_layers:       Option<usize>,
    num_attention_heads:     Option<usize>,
    num_key_value_heads:     Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps:            Option<f64>,
    rope_theta:              Option<f64>,

    bos_token_id:            Option<u32>,
    /// Can be a plain integer or an array of integers (e.g. Llama-3).
    eos_token_id:            Option<serde_json::Value>,

    torch_dtype:             Option<String>,
    tie_word_embeddings:     Option<bool>,
}


#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type:              ModelType,
    pub vocab_size:              usize,
    pub hidden_size:             usize,
    pub intermediate_size:       usize,
    pub num_hidden_layers:       usize,
    pub num_attention_heads:     usize,
    pub num_key_value_heads:     usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps:            f64,
    pub rope_theta:              f64,
    pub bos_token_id:            Option<u32>,
    pub eos_token_id:            Option<u32>,
    pub torch_dtype:             String,
    pub tie_word_embeddings:     bool,
}

impl ModelConfig {
    /// Load from a HuggingFace model directory.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        if !path.exists() {
            return Err(format!("config.json not found in {}", dir.display()));
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read config.json: {e}"))?;
        let raw: RawHfConfig = serde_json::from_str(&text)
            .map_err(|e| format!("cannot parse config.json: {e}"))?;
        Self::from_raw(raw)
    }

    fn from_raw(r: RawHfConfig) -> Result<Self, String> {
        let type_str = r.model_type.clone().unwrap_or_else(|| {
            r.architectures
                .as_deref()
                .and_then(|a| a.first())
                .map(|s| s.to_lowercase())
                .unwrap_or_default()
        });

        let hidden_size         = r.hidden_size.ok_or("config.json: missing hidden_size")?;
        let num_hidden_layers   = r.num_hidden_layers.ok_or("config.json: missing num_hidden_layers")?;
        let num_attention_heads = r.num_attention_heads.ok_or("config.json: missing num_attention_heads")?;
        let vocab_size          = r.vocab_size.ok_or("config.json: missing vocab_size")?;

        let intermediate_size   = r.intermediate_size.unwrap_or(hidden_size * 4);
        let num_key_value_heads = r.num_key_value_heads.unwrap_or(num_attention_heads);
        let max_pos             = r.max_position_embeddings.unwrap_or(4096);
        let rms_norm_eps        = r.rms_norm_eps.unwrap_or(1e-5);
        let rope_theta          = r.rope_theta.unwrap_or(10_000.0);

        let eos_token_id = r.eos_token_id.and_then(|v| match v {
            serde_json::Value::Number(n) => n.as_u64().map(|x| x as u32),
            serde_json::Value::Array(a)  => {
                a.first().and_then(|x| x.as_u64()).map(|x| x as u32)
            }
            _ => None,
        });

        Ok(Self {
            model_type:              ModelType::detect(&type_str),
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings: max_pos,
            rms_norm_eps,
            rope_theta,
            bos_token_id:            r.bos_token_id,
            eos_token_id,
            torch_dtype:             r.torch_dtype.unwrap_or_else(|| "float32".to_string()),
            tie_word_embeddings:     r.tie_word_embeddings.unwrap_or(true),
        })
    }

    pub fn summary(&self) -> String {
        format!(
            "{} | vocab={} | d={} | ffn={} | layers={} | heads={}/{}kv | ctx={}",
            self.model_type,
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> ModelConfig {
        let raw: RawHfConfig = serde_json::from_str(json).unwrap();
        ModelConfig::from_raw(raw).unwrap()
    }

    #[test]
    fn llama_style() {
        let cfg = parse(r#"{
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "torch_dtype": "float16"
        }"#);
        assert_eq!(cfg.model_type, ModelType::Llama);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.eos_token_id, Some(2));
        assert!(cfg.tie_word_embeddings, "should default to true when field is absent");
    }

    #[test]
    fn tied_embeddings_explicit_false() {
        let cfg = parse(r#"{
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "tie_word_embeddings": false
        }"#);
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn tied_embeddings_explicit_true() {
        let cfg = parse(r#"{
            "model_type": "qwen2",
            "vocab_size": 151936,
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "tie_word_embeddings": true
        }"#);
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn eos_as_array() {
        let cfg = parse(r#"{
            "model_type": "llama",
            "vocab_size": 128256,
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "eos_token_id": [128001, 128008, 128009]
        }"#);
        assert_eq!(cfg.eos_token_id, Some(128001));
    }
}
