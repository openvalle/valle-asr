//! Built-in inference backends, and the catalog of models they were validated
//! with. Enable `qwen3` to use the first model family.

mod catalog;
pub use catalog::{ModelFile, ModelSpec, builtin_model, builtin_models};

#[cfg(feature = "qwen3")]
/// Qwen3 inference using locally stored, revision-pinned model weights.
pub mod qwen3;
