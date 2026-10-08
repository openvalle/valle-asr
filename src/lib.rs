//! Local speech recognition with model-independent audio, results, and dispatch.
//!
//! Qwen3-ASR is the first backend. Applications can register additional backends
//! through [`AsrModel`] without changing the result contract or the engine.

mod audio;
#[cfg(feature = "download")]
pub mod cache;
mod engine;
pub mod models;
mod output;
mod streaming;
mod types;

pub use audio::Audio;
pub use engine::{AsrEngine, AsrModel};
pub use output::JsonTranscriptWriter;
pub use streaming::{AudioChunk, WavChunks};
pub use types::{
    ModelInfo, Segment, TimestampMode, TranscribeOptions, Transcript, TranscriptSummary, Word,
};
