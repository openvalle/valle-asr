//! Revision-pinned, SHA-256-verified model downloads shared by every backend.
use crate::{BoxError, Error, Result, error::ensure};
use ring::digest::{Context as Sha256, SHA256};
use serde::Deserialize;
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
    /// Returns [`Error::Invalid`] if no override or usable user directory is
    /// available.
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
        Ok(PathBuf::from(std::env::var_os("HOME").ok_or_else(|| {
            Error::invalid("set VALLE_ASR_CACHE: user cache directory unavailable")
        })?)
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
    /// Returns [`Error::MissingModel`] for missing or corrupt offline files,
    /// [`Error::Download`] for failed downloads and hash mismatches,
    /// [`Error::Io`] for filesystem failures, and [`Error::Invalid`] for unsafe
    /// path components.
    pub fn ensure(&self, spec: &ModelSpec, offline: bool) -> Result<PathBuf> {
        validate_component(&spec.id)?;
        validate_component(&spec.revision)?;
        let root = self.root.join(&spec.id).join(&spec.revision);
        fs::create_dir_all(&root).map_err(Error::io(&root))?;
        let lock_path = root.join(".download.lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(Error::io(&lock_path))?;
        lock.lock().map_err(Error::io(&lock_path))?;
        for file in &spec.files {
            validate_component(&file.path)?;
            let path = root.join(&file.path);
            if verify(&path, file)? {
                continue;
            }
            ensure!(
                !offline,
                Error::MissingModel {
                    id: spec.id.clone(),
                    dir: root,
                }
            );
            ensure!(
                file.sha256.len() == 64 && file.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                Error::invalid(format!("invalid SHA-256 for {}", file.path))
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
            if let Some(source) = error {
                return Err(Error::Download {
                    file: file.path.clone(),
                    source,
                });
            }
        }
        lock.unlock().map_err(Error::io(&lock_path))?;
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
        Error::invalid(format!("unsafe cache path component: {name}"))
    );
    Ok(())
}

fn verify(path: &Path, file: &ModelFile) -> Result<bool> {
    let mut input = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(Error::io(path)(e)),
    };
    if input.metadata().map_err(Error::io(path))?.len() != file.bytes {
        return Ok(false);
    }
    let mut hash = Sha256::new(&SHA256);
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = input.read(&mut buffer).map_err(Error::io(path))?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hex(hash.finish().as_ref()) == file.sha256.to_ascii_lowercase())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// Errors stay in the download's own terms; the caller names the file.
fn download(url: &str, path: &Path, file: &ModelFile) -> Result<(), BoxError> {
    eprintln!("Downloading {} ({} bytes)", file.path, file.bytes);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_global(Some(Duration::from_secs(1800)))
        .user_agent(concat!("valle-asr/", env!("CARGO_PKG_VERSION")))
        .build()
        .into();
    let mut response = agent.get(url).call()?.into_body().into_reader();
    let partial = path.with_extension(format!(
        "{}.part",
        path.extension()
            .and_then(|v| v.to_str())
            .unwrap_or("download")
    ));
    let mut output = File::create(&partial)?;
    let mut hash = Sha256::new(&SHA256);
    let mut bytes = 0u64;
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = response.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        if bytes > file.bytes {
            return Err("server returned more bytes than expected".into());
        }
        output.write_all(&buffer[..n])?;
        hash.update(&buffer[..n]);
    }
    if bytes != file.bytes || hex(hash.finish().as_ref()) != file.sha256.to_ascii_lowercase() {
        return Err("size or SHA-256 mismatch; the partial file will not be used".into());
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
                sha256: hex(ring::digest::digest(&SHA256, b"abc").as_ref()),
            }],
        };
        assert!(matches!(
            cache.ensure(&spec, true),
            Err(Error::MissingModel { .. })
        ));
        let folder = temp.path().join("test/abc");
        fs::write(folder.join("weights.bin.part"), b"abc").unwrap();
        assert!(cache.ensure(&spec, true).is_err());
        fs::write(folder.join("weights.bin"), b"xyz").unwrap();
        assert!(cache.ensure(&spec, true).is_err());
        fs::write(folder.join("weights.bin"), b"abc").unwrap();
        assert_eq!(cache.ensure(&spec, true).unwrap(), folder);
    }
    #[test]
    fn unknown_models_are_named() {
        assert!(matches!(builtin_model("nope"), Err(Error::UnknownModel(id)) if id == "nope"));
        assert!(builtin_model("qwen3-asr-0.6b").is_ok());
    }
    #[test]
    fn prevents_cache_path_traversal() {
        for path in ["../outside", "..", "C:\\temp", "a/b", ""] {
            assert!(validate_component(path).is_err());
        }
    }
}
