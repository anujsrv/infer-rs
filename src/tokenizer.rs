use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;


// Thin wrapper around the HuggingFace `tokenizers` crate's `Tokenizer`.
pub struct Tokenizer {
    inner: HfTokenizer,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
}

impl Tokenizer {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(format!("tokenizer.json not found in {}", dir.display()));
        }
        let inner = HfTokenizer::from_file(&tok_path)
            .map_err(|e| format!("cannot load tokenizer.json: {e}"))?;

        let (bos_id, eos_id) = read_special_ids(dir, &inner);

        Ok(Self { inner, bos_id, eos_id })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.inner
            .encode(text.to_string(), false)
            .expect("tokenizer encode failed");
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner
            .decode(&ids, false)
            .expect("tokenizer decode failed")
    }

    pub fn decode_token(&self, id: usize) -> String {
        self.decode(&[id])
    }

    pub fn bos_token_id(&self) -> Option<u32> { self.bos_id }
    pub fn eos_token_id(&self) -> Option<u32> { self.eos_id }
}


fn read_special_ids(dir: &Path, inner: &HfTokenizer) -> (Option<u32>, Option<u32>) {
    let cfg_path = dir.join("tokenizer_config.json");
    if !cfg_path.exists() {
        return (None, None);
    }
    let text = match std::fs::read_to_string(&cfg_path) {
        Ok(t) => t,
        Err(_) => return (None, None),
    };
    let cfg: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return (None, None),
    };
    (
        special_id(&cfg, "bos_token", inner),
        special_id(&cfg, "eos_token", inner),
    )
}

fn special_id(cfg: &serde_json::Value, field: &str, inner: &HfTokenizer) -> Option<u32> {
    let v = cfg.get(field)?;
    let content = match v {
        serde_json::Value::String(s) => s.as_str(),
        serde_json::Value::Object(o) => o.get("content").and_then(|c| c.as_str())?,
        _ => return None,
    };
    inner.token_to_id(content)
}
