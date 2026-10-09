//! Revision-pinned, SHA-256-verified model downloads shared by every backend.
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

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
/// A downloadable model revision and all artifacts needed to load it.
pub struct ModelSpec {
    /// Unique model ID used for selection and cache layout.
    pub id: String,
    /// Hugging Face repository in `owner/name` form.
    pub repository: String,
    /// Pinned upstream commit or immutable revision.
    pub revision: String,
    /// License identifier supplied by the model catalog.
    pub license: String,
    /// Files verified before the revision is returned to the caller.
    pub files: Vec<ModelFile>,
}

/// Return the revision-pinned model catalog embedded in this crate.
pub fn builtin_models() -> Vec<ModelSpec> {
    serde_json::from_str(include_str!("../models.json")).expect("embedded model catalog is valid")
}

/// Find a built-in model by ID. Returns an error for an unknown ID.
pub fn builtin_model(id: &str) -> Result<ModelSpec> {
    builtin_models()
        .into_iter()
        .find(|s| s.id == id)
        .with_context(|| format!("unknown model: {id}"))
}

/// On-disk, cross-process-locked model storage with verified downloads.
pub struct ModelCache {
    root: PathBuf,
}
impl ModelCache {
    /// Select a cache root; directories are created when [`Self::ensure`] runs.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    /// Resolve `VALLE_ASR_CACHE`, then the platform user cache directory.
    ///
    /// # Errors
    /// Returns an error if no override or usable user directory is available.
    pub fn default_path() -> Result<PathBuf> {
        if let Some(path) = std::env::var_os("VALLE_ASR_CACHE") {
            return Ok(path.into());
        }
        #[cfg(target_os = "windows")]
        if let Some(path) = std::env::var_os("LOCALAPPDATA") {
            return Ok(PathBuf::from(path).join("valle-asr").join("models"));
        }
        #[cfg(target_os = "macos")]
        if let Some(path) = std::env::var_os("HOME") {
            return Ok(PathBuf::from(path).join("Library/Caches/valle-asr/models"));
        }
        if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
            return Ok(PathBuf::from(path).join("valle-asr/models"));
        }
        Ok(PathBuf::from(
            std::env::var_os("HOME")
                .context("set VALLE_ASR_CACHE: user cache directory unavailable")?,
        )
        .join(".cache/valle-asr/models"))
    }
    /// Validate every file even on a cache hit. Offline never uses corrupt or
    /// incomplete artifacts. Per-revision file locks serialize concurrent writers.
    ///
    /// Even offline calls create the revision directory and writer lock, so
    /// the cache root must be writable. Model files must remain immutable while
    /// loaded by an inference backend.
    ///
    /// # Errors
    /// Returns an error for unsafe path components, invalid or missing offline
    /// artifacts, failed downloads, hash mismatches, or filesystem failures.
    pub fn ensure(&self, spec: &ModelSpec, offline: bool) -> Result<PathBuf> {
        validate_component(&spec.id)?;
        validate_component(&spec.revision)?;
        let root = self.root.join(&spec.id).join(&spec.revision);
        fs::create_dir_all(&root)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join(".download.lock"))?;
        fs2::FileExt::lock_exclusive(&lock)?;
        for file in &spec.files {
            validate_component(&file.path)?;
            let path = root.join(&file.path);
            if verify(&path, file)? {
                continue;
            }
            ensure!(
                !offline,
                "model file missing or corrupt in offline cache: {}",
                path.display()
            );
            ensure!(
                file.sha256.len() == 64 && file.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid file hash"
            );
            let url = format!(
                "https://huggingface.co/{}/resolve/{}/{}?download=true",
                spec.repository, spec.revision, file.path
            );
            let mut error = None;
            for attempt in 0..3 {
                match download(&url, &path, file) {
                    Ok(()) => {
                        error = None;
                        break;
                    }
                    Err(e) => {
                        error = Some(e);
                        if attempt < 2 {
                            std::thread::sleep(Duration::from_secs(2 * (attempt + 1)));
                        }
                    }
                }
            }
            if let Some(error) = error {
                return Err(error).with_context(|| format!("download {}", file.path));
            }
        }
        fs2::FileExt::unlock(&lock)?;
        Ok(root)
    }
}

fn validate_component(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name != "."
            && name != ".."
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b)),
        "unsafe cache path component: {name}"
    );
    Ok(())
}

fn verify(path: &Path, file: &ModelFile) -> Result<bool> {
    let mut input = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e.into()),
    };
    if input.metadata()?.len() != file.bytes {
        return Ok(false);
    }
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hex(&hash.finalize()) == file.sha256.to_ascii_lowercase())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn download(url: &str, path: &Path, file: &ModelFile) -> Result<()> {
    eprintln!("Downloading {} ({} bytes)", file.path, file.bytes);
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(1800))
        .user_agent("valle-asr/0.1")
        .build()?;
    let mut response = client.get(url).send()?.error_for_status()?;
    let partial = path.with_extension(format!(
        "{}.part",
        path.extension()
            .and_then(|v| v.to_str())
            .unwrap_or("download")
    ));
    let mut output = File::create(&partial)?;
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = response.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        ensure!(
            bytes <= file.bytes,
            "server returned too many bytes for {}",
            file.path
        );
        output.write_all(&buffer[..n])?;
        hash.update(&buffer[..n]);
    }
    if bytes != file.bytes || hex(&hash.finalize()) != file.sha256.to_ascii_lowercase() {
        bail!(
            "size/SHA-256 mismatch for {}; partial file will not be used",
            file.path
        );
    }
    output.sync_all()?;
    drop(output);
    // Windows rename cannot replace an existing destination. Only invalid files
    // reach this branch; a valid revision is never overwritten.
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(&partial, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offline_cache_rejects_corruption_and_partial_downloads() {
        let temp = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(temp.path());
        let spec = ModelSpec {
            id: "test".into(),
            repository: "test/test".into(),
            revision: "abc".into(),
            license: "MIT".into(),
            files: vec![ModelFile {
                path: "weights.bin".into(),
                bytes: 3,
                sha256: hex(&Sha256::digest(b"abc")),
            }],
        };
        assert!(cache.ensure(&spec, true).is_err());
        let folder = temp.path().join("test/abc");
        fs::write(folder.join("weights.bin.part"), b"abc").unwrap();
        assert!(cache.ensure(&spec, true).is_err());
        fs::write(folder.join("weights.bin"), b"xyz").unwrap();
        assert!(cache.ensure(&spec, true).is_err());
        fs::write(folder.join("weights.bin"), b"abc").unwrap();
        assert_eq!(cache.ensure(&spec, true).unwrap(), folder);
    }
    #[test]
    fn prevents_cache_path_traversal() {
        for path in ["../outside", "..", "C:\\temp", "a/b", ""] {
            assert!(validate_component(path).is_err());
        }
    }
}
