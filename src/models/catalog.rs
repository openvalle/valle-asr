use crate::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
/// One immutable model artifact with its expected size and SHA-256 digest.
pub struct ModelFile {
    /// Single file name relative to the model revision directory.
    pub path: String,
    /// Expected artifact size in bytes.
    pub bytes: u64,
    /// Hexadecimal SHA-256 digest of the complete file.
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
/// A model revision the backends were validated with, and every file it needs.
pub struct ModelSpec {
    /// Unique model ID used for selection.
    pub id: String,
    /// Hugging Face repository in `owner/name` form.
    pub repository: String,
    /// Pinned upstream commit or immutable revision.
    pub revision: String,
    /// License identifier supplied by the model catalog.
    pub license: String,
    /// Files a model directory must hold.
    pub files: Vec<ModelFile>,
}

impl ModelSpec {
    /// Files that `dir` lacks or holds with a different size. An empty list
    /// means the directory is complete.
    ///
    /// This reads no file contents, so it is fast enough to call before every
    /// load. It does not check SHA-256: a downloader should check each file's
    /// hash before giving it its final name, so a file of the right size and
    /// name was already verified when it was written.
    pub fn missing(&self, dir: impl AsRef<Path>) -> Vec<&ModelFile> {
        let dir = dir.as_ref();
        self.files
            .iter()
            .filter(|file| {
                std::fs::metadata(dir.join(&file.path))
                    .map_or(true, |m| !m.is_file() || m.len() != file.bytes)
            })
            .collect()
    }
}

/// Return the revision-pinned model catalog embedded in this crate. It is data
/// only; this crate downloads nothing.
pub fn builtin_models() -> Vec<ModelSpec> {
    serde_json::from_str(include_str!("../../models.json"))
        .expect("embedded model catalog is valid")
}

/// Find a built-in model by ID.
///
/// # Errors
/// Returns [`Error::UnknownModel`] for an ID not in the catalog.
pub fn builtin_model(id: &str) -> Result<ModelSpec> {
    builtin_models()
        .into_iter()
        .find(|s| s.id == id)
        .ok_or_else(|| Error::UnknownModel(id.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_models_are_named() {
        assert!(matches!(builtin_model("nope"), Err(Error::UnknownModel(id)) if id == "nope"));
        assert!(builtin_model("qwen3-asr-0.6b").is_ok());
    }

    #[test]
    fn missing_lists_absent_and_wrong_sized_files() {
        let dir = tempfile::tempdir().unwrap();
        let file = |path: &str, bytes| ModelFile {
            path: path.into(),
            bytes,
            sha256: String::new(),
        };
        let spec = ModelSpec {
            id: "test".into(),
            repository: "test/test".into(),
            revision: "abc".into(),
            license: "MIT".into(),
            files: vec![file("a.bin", 3), file("b.bin", 2)],
        };
        let names = |spec: &ModelSpec| {
            spec.missing(dir.path())
                .iter()
                .map(|f| f.path.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&spec), ["a.bin", "b.bin"]);
        std::fs::write(dir.path().join("a.bin"), b"abc").unwrap();
        std::fs::write(dir.path().join("b.bin"), b"abc").unwrap();
        assert_eq!(names(&spec), ["b.bin"]);
        std::fs::write(dir.path().join("b.bin"), b"ab").unwrap();
        assert!(names(&spec).is_empty());
        std::fs::create_dir(dir.path().join("c.bin")).unwrap();
        let spec = ModelSpec {
            files: vec![file("c.bin", 0)],
            ..spec
        };
        assert_eq!(names(&spec), ["c.bin"]);
    }
}
