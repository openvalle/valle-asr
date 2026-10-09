use crate::CancellationToken;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// One aligned word, or one Han character for Chinese/Cantonese.
pub struct Word {
    /// Recognized text for this item.
    pub text: String,
    /// Inclusive start on the source timeline, in milliseconds.
    pub start_ms: u64,
    /// End on the source timeline, in milliseconds.
    pub end_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Text and word spans for a source interval.
pub struct Segment {
    /// Recognized text for this item.
    pub text: String,
    /// Detected language name for this result.
    pub language: String,
    /// Inclusive start on the source timeline, in milliseconds.
    pub start_ms: u64,
    /// End on the source timeline, in milliseconds.
    pub end_ms: u64,
    /// Aligned word spans; empty when word timestamps are disabled.
    pub words: Vec<Word>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// An in-memory transcription; use the file API to bound result memory.
pub struct Transcript {
    /// ID of the backend that produced the result.
    pub model: String,
    /// Recognized text for this item.
    pub text: String,
    /// Detected language name for this result.
    pub language: String,
    /// Total source duration in milliseconds.
    pub duration_ms: u64,
    /// Source intervals retained in memory.
    pub segments: Vec<Segment>,
}

/// Metadata returned by file transcription; segments are delivered to a sink.
/// It deliberately contains no growing transcript text or segment vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptSummary {
    /// ID of the backend that produced the result.
    pub model: String,
    /// Detected language name for this result.
    pub language: String,
    /// Total source duration in milliseconds.
    pub duration_ms: u64,
    /// Number of intervals emitted to the sink.
    pub segment_count: u64,
}

#[derive(Debug, Clone, Serialize)]
/// Model identity and timestamp capabilities reported by a backend.
pub struct ModelInfo {
    /// Unique model ID used for engine dispatch.
    pub id: String,
    /// Model family name.
    pub family: String,
    /// Whether the backend can return learned word timestamps.
    pub word_timestamps: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Select whether recognition also performs learned word alignment.
pub enum TimestampMode {
    /// Return transcription without word alignment.
    None,
    #[default]
    /// Align words to source times; requires an alignment-capable backend.
    Word,
}

#[derive(Debug, Clone)]
/// Per-request recognition controls; defaults select 30-second chunks and words.
pub struct TranscribeOptions {
    /// English language name or an ISO code; None requests language detection.
    pub language: Option<String>,
    /// Requested timestamp granularity; defaults to [`TimestampMode::Word`].
    pub timestamps: TimestampMode,
    /// Maximum audio length per inference pass; supported range is 1..=30 s.
    pub chunk_seconds: u32,
    /// Positive generated-text token limit per chunk; defaults to 448.
    pub max_new_tokens: usize,
    /// Optional domain/context text included in the system prompt.
    pub context: String,
    /// Shared stop signal, checked during decoding, inference and word alignment.
    /// Cancellation returns [`crate::Cancelled`]; create a new token for each request.
    pub cancellation: CancellationToken,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            timestamps: TimestampMode::Word,
            chunk_seconds: 30,
            max_new_tokens: 448,
            context: String::new(),
            cancellation: CancellationToken::default(),
        }
    }
}

impl TranscribeOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        self.cancellation.check()?;
        ensure!(
            (1..=30).contains(&self.chunk_seconds),
            "chunk_seconds must be 1..=30"
        );
        ensure!(self.max_new_tokens > 0, "max_new_tokens must be positive");
        Ok(())
    }
}
