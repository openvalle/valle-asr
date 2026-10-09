//! Built-in inference backends. Enable `qwen3` to use the first model family.

#[cfg(feature = "qwen3")]
/// Qwen3 inference using locally stored, revision-pinned model weights.
pub mod qwen3;
