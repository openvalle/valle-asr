use std::{fmt, path::PathBuf};

/// An error from an audio source or a segment sink, kept as the caller made it.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// The result of every fallible call in this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Why a request failed. Match on the variant; the message is for people.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The request's [`crate::CancellationToken`] was cancelled. No partial
    /// result is returned.
    #[error("operation cancelled")]
    Cancelled,
    /// A model ID that is not in the built-in catalog, or not registered with
    /// an [`crate::AsrEngine`].
    #[error("unknown model: {0}")]
    UnknownModel(String),
    /// A language the model does not recognize.
    #[error("unsupported language: {0}")]
    Language(String),
    /// Invalid options, samples or arguments.
    #[error("{0}")]
    Invalid(String),
    /// The audio source failed to read.
    #[error("read audio")]
    Audio(#[source] BoxError),
    /// The segment sink returned an error.
    #[error("segment sink")]
    Sink(#[source] BoxError),
    /// A model directory could not be loaded, or inference failed.
    #[error("model: {0}")]
    Model(String),
    /// A filesystem operation failed.
    #[error("{}", path.display())]
    Io {
        /// File or directory involved.
        path: PathBuf,
        /// What failed.
        #[source]
        source: std::io::Error,
    },
}

impl Error {
    pub(crate) fn invalid(message: impl fmt::Display) -> Self {
        Self::Invalid(message.to_string())
    }

    pub(crate) fn model(message: impl fmt::Display) -> Self {
        Self::Model(message.to_string())
    }

    #[cfg(feature = "qwen3")]
    pub(crate) fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Self {
        let path = path.into();
        move |source| Self::Io { path, source }
    }
}

#[cfg(feature = "qwen3")]
impl From<candle_core::Error> for Error {
    fn from(error: candle_core::Error) -> Self {
        Self::model(error)
    }
}

/// Return `error` unless `condition` holds.
macro_rules! ensure {
    ($condition:expr, $error:expr) => {
        if !$condition {
            return Err($error);
        }
    };
}
pub(crate) use ensure;
