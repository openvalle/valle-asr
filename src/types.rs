use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Word {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    pub text: String,
    pub language: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub words: Vec<Word>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Transcript {
    pub model: String,
    pub text: String,
    pub language: String,
    pub duration_ms: u64,
    pub segments: Vec<Segment>,
}

/// Metadata returned by file transcription; segments are delivered to a sink.
/// It deliberately contains no growing transcript text or segment vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptSummary {
    pub model: String,
    pub language: String,
    pub duration_ms: u64,
    pub segment_count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub family: String,
    pub word_timestamps: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TimestampMode {
    None,
    #[default]
    Word,
}

#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// English language name or an ISO code; None requests language detection.
    pub language: Option<String>,
    pub timestamps: TimestampMode,
    /// Maximum audio length per inference pass; supported range is 1..=30 s.
    pub chunk_seconds: u32,
    pub max_new_tokens: usize,
    /// Optional domain/context text included in the system prompt.
    pub context: String,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            timestamps: TimestampMode::Word,
            chunk_seconds: 30,
            max_new_tokens: 448,
            context: String::new(),
        }
    }
}
