#![doc = include_str!("../README.md")]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

mod audio;
mod cancellation;
#[cfg(feature = "qwen3")]
mod chunks;
mod engine;
mod error;
/// Built-in optional model backends.
pub mod models;
mod types;

pub use audio::{AudioSource, SAMPLE_RATE};
pub use cancellation::CancellationToken;
pub use engine::{AsrEngine, AsrModel};
pub use error::{BoxError, Error, Result};
pub use types::{
    ModelInfo, Segment, TimestampMode, TranscribeOptions, Transcript, TranscriptSummary, Word,
};
