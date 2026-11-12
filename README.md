# infer-rs

A small Rust implementation of inference engine, using Candle for tensor operations and Hugging Face tokenizers. The transformer layers live in this repo rather than using a prebuilt model implementation.

## Run it

You'll need Rust and a local copy of [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main). Download `config.json`, `tokenizer.json`, and `model.safetensors` into `models/Qwen2.5-0.5B-Instruct/`, then run:

```sh
cargo run --release -- --model models/Qwen2.5-0.5B-Instruct "The capital of France is"
```

It prints the generated text and a tokens-per-second summary.

## Current limits

CPU-only, float32, greedy decoding, up to 128 new tokens. There's no KV cache, so each step processes the whole sequence again. Prompts are passed through as-is, chat templates aren't applied automatically.

The model loader currently targets Qwen2-style models with tied embeddings, not every architecture recognized by the config parser.

## Tests

```sh
cargo test
```
