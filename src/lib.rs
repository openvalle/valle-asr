#![doc = include_str!("../README.md")]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

mod audio;
#[cfg(feature = "download")]
pub mod cache;
mod engine;
/// Built-in optional model backends.
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
