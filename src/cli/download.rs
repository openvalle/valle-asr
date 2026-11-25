//! Revision-pinned, SHA-256-verified model downloads for the command line and CI.
//! The library only describes models; fetching them is the caller's job.
use crate::Result;
use ring::digest::{Context as Sha256, SHA256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use valle_asr::models::{ModelFile, ModelSpec};

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
                .ok_or("set VALLE_ASR_CACHE: user cache directory unavailable")?,
        )
        .join(".cache/valle-asr/models"))
    }
    /// Return the revision directory, downloading files that are missing or
    /// have the wrong size. A download is hashed before it gets its final name,
    /// so a cache hit checks sizes only; `rehash` also hashes every cached file.
    /// Offline never uses incomplete artifacts. Per-revision file locks
    /// serialize concurrent writers. Even offline calls create the revision
    /// directory and writer lock, so the cache root must be writable.
    pub fn ensure(&self, spec: &ModelSpec, offline: bool, rehash: bool) -> Result<PathBuf> {
        validate_component(&spec.id)?;
        validate_component(&spec.revision)?;
        let root = self.root.join(&spec.id).join(&spec.revision);
        fs::create_dir_all(&root).map_err(|e| format!("{}: {e}", root.display()))?;
        let lock_path = root.join(".download.lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)?;
        lock.lock()?;
        for file in &spec.files {
            validate_component(&file.path)?;
            let path = root.join(&file.path);
            if verify(&path, file, rehash)? {
                continue;
            }
            ensure!(
                !offline,
                "model {} is missing or corrupt in the offline cache at {}",
                spec.id,
                root.display()
            );
            ensure!(
                file.sha256.len() == 64 && file.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid SHA-256 for {}",
                file.path
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
                return Err(format!("download {}: {error}", file.path).into());
            }
        }
        lock.unlock()?;
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

fn verify(path: &Path, file: &ModelFile, rehash: bool) -> Result<bool> {
    let mut input = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(format!("{}: {e}", path.display()).into()),
    };
    if input.metadata()?.len() != file.bytes {
        return Ok(false);
    }
    if !rehash {
        return Ok(true);
    }
    let mut hash = Sha256::new(&SHA256);
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = input.read(&mut buffer)?;
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
fn download(url: &str, path: &Path, file: &ModelFile) -> Result<()> {
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
        assert!(cache.ensure(&spec, true, false).is_err());
        let folder = temp.path().join("test/abc");
        fs::write(folder.join("weights.bin.part"), b"abc").unwrap();
        assert!(cache.ensure(&spec, true, false).is_err());
        fs::write(folder.join("weights.bin"), b"ab").unwrap();
        assert!(cache.ensure(&spec, true, false).is_err());
        // Same size, wrong content: only a rehash notices.
        fs::write(folder.join("weights.bin"), b"xyz").unwrap();
        assert_eq!(cache.ensure(&spec, true, false).unwrap(), folder);
        assert!(cache.ensure(&spec, true, true).is_err());
        fs::write(folder.join("weights.bin"), b"abc").unwrap();
        assert_eq!(cache.ensure(&spec, true, true).unwrap(), folder);
    }
    #[test]
    fn prevents_cache_path_traversal() {
        for path in ["../outside", "..", "C:\\temp", "a/b", ""] {
            assert!(validate_component(path).is_err());
        }
    }
}
