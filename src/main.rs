mod config;
mod model;
mod tokenizer;

use std::io::Write;
use std::path::PathBuf;

use candle_core::{DType, Device};

use config::ModelConfig;
use tokenizer::Tokenizer;
use crate::model::qwen2::Qwen2Model;

const MAX_NEW_TOKENS: usize = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (model_dir, prompt) = if args.len() >= 4 && args[1] == "--model" {
        (PathBuf::from(&args[2]), args[3].clone())
    } else {
        eprintln!("Usage: {} --model <dir> \"<prompt>\"", args[0]);
        std::process::exit(1);
    };

    let config    = ModelConfig::from_dir(&model_dir)?;
    let tokenizer = Tokenizer::from_dir(&model_dir)?;
    let device    = Device::Cpu;
    let model     = Qwen2Model::load(&model_dir, &device, DType::F32, &config)?;

    println!("Model:  {}", config.summary());
    println!("Prompt: {prompt:?}");

    let mut tokens: Vec<u32> = tokenizer
        .encode(&prompt)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let prompt_len = tokens.len();
    // Qwen2.5 emits `<|im_end|>` (151645) at the end of an assistant turn;
    // 151643 is `<|endoftext|>` (general EOS). Either is a reasonable stop.
    let eos = config.eos_token_id.unwrap_or(151643);
    println!("Prompt tokens: {prompt_len}");

    print!("\n{prompt}");
    std::io::stdout().flush()?;
    let start = std::time::Instant::now();
    let mut stopped_by_eos = false;

    for _ in 0..MAX_NEW_TOKENS {
        let logits = model.forward(&tokens)?;           // [seq, vocab]
        let last   = logits.get(tokens.len() - 1)?;     // [vocab]
        let next   = last.argmax(0)?.to_scalar::<u32>()?;

        if next == eos {
            stopped_by_eos = true;
            break;
        }

        let piece = tokenizer.decode_token(next as usize);
        print!("{piece}");
        std::io::stdout().flush()?;

        tokens.push(next);
    }
    let elapsed = start.elapsed();

    let gen_n = tokens.len() - prompt_len;
    println!(
        "\n\n[generated {gen_n} tokens in {:.1}s ({:.2} tok/s){}]",
        elapsed.as_secs_f64(),
        gen_n as f64 / elapsed.as_secs_f64().max(1e-6),
        if stopped_by_eos { ", stopped on EOS" } else { ", hit max_new_tokens" },
    );

    Ok(())
}
