# valle-asr

Local speech recognition in Rust with selectable model backends and learned
word timestamps. The first backend is **Qwen3-ASR**, paired with
**Qwen3-ForcedAligner**. This repository is independent of `valle`.

[![Linux](https://github.com/openvalle/valle-asr/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/openvalle/valle-asr/actions/workflows/ci-linux.yml)
[![macOS](https://github.com/openvalle/valle-asr/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/openvalle/valle-asr/actions/workflows/ci-macos.yml)
[![Windows](https://github.com/openvalle/valle-asr/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/openvalle/valle-asr/actions/workflows/ci-windows.yml)

## Scope

- Rust **1.99.0**, edition 2024; direct crates use the latest stable releases
  checked on 2026-11-24, with an exact `Cargo.lock`.
- Native CPU inference on Windows, Linux and macOS, each on x86_64 and ARM64,
  through Candle. On macOS, matrix products use the system Accelerate
  framework. Python, libtorch, OpenBLAS and a GPU are not required to
  transcribe audio.
- A catalog of the official Qwen3-ASR 0.6B and ForcedAligner 0.6B revisions it
  was validated with, and an offline check of a model directory against it.
  The library downloads nothing; the CLI's downloader fetches the catalog's
  files for development and CI. Local directories can also load the matching
  original Qwen3-ASR 1.7B architecture; 1.7B has not yet been included in the
  validation matrix.
- The library takes mono 16 kHz f32 samples and returns segments with word
  timestamps. Decoding media is left to the caller, such as Valle's FFmpeg
  integration. The CLI reads mono/stereo integer/float WAV and resamples it
  to 16 kHz with a low-pass filter.
- Audio of any length is transcribed in bounded chunks with silence-aware
  boundaries. A source that reads samples on demand keeps audio RAM from
  growing with duration. Word timestamps keep the original timeline.
- Pluggable model interface: applications register models and select by ID.
  Adding a model family does not change the shared transcript schema.

## Run

```sh
cargo run --release -- models
cargo run --release -- download --cache-dir models
cargo run --release -- transcribe speech.wav --cache-dir models --language zh
```

Word timestamps are enabled by default. Use `--text-only` to run only ASR, and
`--offline` to require previously downloaded, verified weights. To bypass the
downloader and use existing official safetensors directories:

```sh
cargo run --release -- transcribe speech.wav \
  --model-dir /path/to/Qwen3-ASR-0.6B \
  --aligner-dir /path/to/Qwen3-ForcedAligner-0.6B \
  --language en --output transcript.json
```

On Windows, use Windows paths and put the command on a single line in
PowerShell. No WSL is involved. Additional options include `--chunk-seconds`
(1–30), `--max-new-tokens`, and `--context` for domain vocabulary.

Results contain model ID, language, full text, original duration, segments,
and per-word `{text, start_ms, end_ms}`. Milliseconds are absolute source times.
For Chinese and Cantonese, a timestamp unit is one Han character, matching the
official processor. Word alignment currently supports Chinese, Cantonese,
English, French, German, Spanish, Italian, Portuguese and Russian. Japanese
and Korean morphological word segmentation is not implemented; those
languages can use text-only ASR. Other ASR languages unsupported by the
aligner return an explicit error when word timestamps are requested.

## Long files and memory

Submit a complete file with the same `transcribe ... --output transcript.json`
command. There is no duration limit. The library reads one chunk at a time from
an `AudioSource`. A 30-second 16 kHz chunk contains 1,920,000 bytes of F32 PCM,
regardless of duration; model weights and inference tensors need additional
RAM. The CLI's WAV source reads at most the requested range plus resampling
neighbors, mixes channels during reading, and keeps the resampling phase across
boundaries. It supports input sample rates up to 384 kHz.

Within the last three seconds of a chunk (or the last quarter of a small chunk),
chunking prefers a quiet interval of at least 120 ms. It does not drop or
duplicate source samples. Continuous speech without a suitable pause still
uses the hard chunk limit; this is not a full VAD or overlap/deduplication system.
Exact digital-zero padding is omitted from neural inference while preserving
its offset. Other low-volume sounds are not classified as silence.

With word timestamps, ASR runs first and keeps each chunk's interval and text,
a few hundred bytes per chunk. Its neural weights are released before the
alignment pass reads the same samples again. Each neural model loads once per stage. Final segments
are emitted during alignment; text-only mode emits them during the first pass.
Mapped BF16 weights are converted directly into owned CPU F32 storage without
an intermediate BF16 tensor. Each tensor uses a short-lived mapping, so whole
checkpoint mappings do not remain resident alongside the final F32 weights.
Tokenizers are shared across chunks. Alignment
retains the complete attention context but projects only timestamp positions
and releases each layer's temporary KV after use; ASR keeps its decode KV.
The summary keeps the full text for the transcript's `text` field. The CLI
writes each segment to its JSON output as it arrives. Processing time grows
with duration. `--output` replaces the destination only after successful
completion.

Tokenizers remain in memory. A bounded resident cache holds one neural model:
all ASR chunks run before the weights are released and all alignment chunks
run. Repeated text-only transcription and standalone alignment reuse their
resident model; full transcription switches models once per stage. This
avoids retaining both sets of F32 weights on small CI runners. `Qwen3::unload`
releases the resident model explicitly.

The 0.6B models' F32 weights take 2.9 GiB for ASR and 3.4 GiB for the aligner.
On macOS arm64 the CLI peaks at about 3.5 GB for text only and 4.9 GB with
word timestamps. The difference beyond the aligner's weights is freed ASR
memory that the system allocator keeps for reuse; it is not leaked.

The CLI's WAV reader supports standard RIFF WAV, whose 32-bit size fields impose
an approximately 4 GiB container limit (about 37 hours for 16 kHz mono PCM16).
RF64 and compressed media are not decoded here.

## Rust API

```toml
[dependencies]
valle-asr = { version = "0.3.0", default-features = false, features = ["qwen3"] }
```

The library reads models from local directories and never uses the network.
`valle_asr::models::builtin_models()` lists the revisions it was validated
with: each one's Hugging Face repository and commit, and every file's size and
SHA-256. A downloader can fetch and verify those files, then call
`ModelSpec::missing(dir)` before loading to see which files a directory lacks
or holds with the wrong size. That check reads no file contents, so a
downloader should verify each file's SHA-256 before giving it its final name.

```rust,no_run
# #[cfg(feature = "qwen3")]
# fn main() -> valle_asr::Result<()> {
use valle_asr::{AsrEngine, TranscribeOptions, models::qwen3::Qwen3};

let mut engine = AsrEngine::new();
engine.register(Qwen3::load(
    "qwen3-asr-0.6b",
    "models/asr",
    Some("models/aligner".into()),
)?)?;
// Mono 16 kHz samples in [-1, 1], decoded by the application.
let samples: Vec<f32> = vec![0.0; 16_000];
let transcript = engine.transcribe("qwen3-asr-0.6b", &samples, &TranscribeOptions::default())?;
println!("{}", transcript.text);
# Ok(())
# }
# #[cfg(not(feature = "qwen3"))]
# fn main() {}
```

`transcribe` keeps every segment in memory. For long audio, implement
`AudioSource` to read samples on demand, for example from a decoder that can
seek, and call `transcribe_source` with a sink that persists or displays each
segment. Word timestamps read every chunk twice, once for recognition and once
for alignment, so a source must allow reads from any offset. A `&[f32]` is a
source as well. A sink that collects every segment into a vector still grows RAM.

Implement `AsrModel` to register another model family. `AsrEngine` dispatches
to the selected registered instance and checks that segments arrive in order
inside the audio. Backends read the same samples and return the same
model-independent result types. Qwen is optional:
`cargo check --no-default-features --lib` builds the core API alone.

### Errors

Every fallible call returns `valle_asr::Result<T>`, whose error is the
`valle_asr::Error` enum. Match on its variants: `Cancelled`, `UnknownModel`,
`Language`, `Invalid` (options, samples or arguments), `Audio` and `Sink` (holding the
error your source or sink returned), `Model` (loading or inference) and `Io`.
The enum is non-exhaustive. Sources and sinks return
`Result<_, valle_asr::BoxError>`, so they can fail with their own error types.

### Cancellation

Keep a clone of the request's `CancellationToken` in the UI or controlling
thread, and call `cancel()` to stop work:

```rust
use valle_asr::{CancellationToken, TranscribeOptions};

let stop = CancellationToken::default();
let options = TranscribeOptions {
    cancellation: stop.clone(),
    ..Default::default()
};
// Pass &options to the inference worker; keep stop in the controlling thread.
stop.cancel();
assert!(options.cancellation.is_cancelled());
```

`CancellationToken::from_shared_flag(Arc<AtomicBool>)` also accepts a host's
existing stop flag without a monitoring thread. The flag must only transition
from `false` to `true` during a request. Cancellation is permanent for all token
clones; use a fresh token for the next request.

Qwen checks cancellation between chunks, converting
weights, computing mel frames, running encoder/decoder/classifier layers and
generating tokens. Both ASR and alignment passes check the flag, so word
transcription can stop before its first result callback. Standalone alignment
can use `Qwen3::align_with_cancellation`; `align` retains its original behavior.

A cancelled request returns `Error::Cancelled` and never reports successful
partial output. Already delivered sink chunks
cannot be retracted. Call a transactional output writer's `finish()` only after
successful inference, and reuse the backend with a fresh token after cancellation.
Custom backends must check the token during their own compute; the engine also
checks before dispatch, around sink callbacks and before returning success.

Cancellation is cooperative: a running tensor/CPU operator or blocking file
operation finishes before the next checkpoint. Model constructors have no
cancellation parameter. No fixed wall-clock cancellation latency is
promised, and no extra polling thread is created by the library.

### Features

- `qwen3`: the built-in Qwen3 CPU backend.
- `cli`: the command-line application with its WAV reader, JSON output and
  model downloader, including `qwen3`.

Both are enabled by default. The model-independent core can be used with
`default-features = false` and no additional features.

## CLI model cache

The CLI's downloader serves development and CI; it fetches from Hugging Face
only, without resuming an interrupted file. Applications should fetch models
themselves. The CLI's cache defaults to `VALLE_ASR_CACHE` when set, otherwise
to the native user cache directory. `--cache-dir` overrides it. Every revision has its own
directory and cross-process writer lock. Existing files are reused only after
size and SHA-256 validation; interrupted `.part` files are never treated as
models. Downloads are verified before a final rename. Revisions are immutable
while model files are mapped. Do not modify a directory being used for inference.

GitHub Actions caches Cargo dependencies/builds per OS and CPU architecture
and shares the revision-pinned model cache across platforms. Model cache keys include the
hash of `models.json`; each restored cache is still validated before use.
Text files use LF on every platform so Windows checkout does not change
content-based cache keys; WAV fixtures remain binary.

## Validation

| OS | CPU | Rust target | GitHub runner |
|---|---|---|---|
| Linux | x86_64 | `x86_64-unknown-linux-gnu` | [`ubuntu-24.04`](https://github.com/openvalle/valle-asr/actions/workflows/ci-linux.yml) |
| Linux | arm64 | `aarch64-unknown-linux-gnu` | [`ubuntu-24.04-arm`](https://github.com/openvalle/valle-asr/actions/workflows/ci-linux-arm64.yml) |
| Windows | x86_64 | `x86_64-pc-windows-msvc` | [`windows-2022`](https://github.com/openvalle/valle-asr/actions/workflows/ci-windows.yml) |
| Windows | arm64 | `aarch64-pc-windows-msvc` | [`windows-11-arm`](https://github.com/openvalle/valle-asr/actions/workflows/ci-windows-arm64.yml) |
| macOS | arm64 | `aarch64-apple-darwin` | [`macos-15`](https://github.com/openvalle/valle-asr/actions/workflows/ci-macos.yml) |
| macOS | x86_64 | `x86_64-apple-darwin` | [`macos-15-intel`](https://github.com/openvalle/valle-asr/actions/workflows/ci-macos-x86_64.yml) |

Each Linux, macOS and Windows x86_64/ARM64 combination has an
**independent native workflow**, with all six triggered in parallel. CI checks
both the runner CPU and Rust host triple before compilation. Each runs
formatting, Clippy, the permissive-license gate, optional feature compilation, API/unit tests, then actual 0.6B transcription and word
alignment. Missing weights fail the model test.

The real tests check English/Chinese character error rates, non-collapsed
monotonic word spans inside source chunks, and offsets on a two-chunk clip.
Every platform also streams a two-hour sparse source through the engine, runs
the WAV reader's continuous resampling tests, and runs real Qwen ASR/alignment
on speech at both ends of a two-hour sparse source. The long fixture is mostly digital silence;
it validates file handling, bounded chunks and global word offsets, not two
hours of continuous-speech accuracy or throughput.
Each platform uploads the output JSON and a timing/accuracy receipt. They
validate correctness on hosted CPUs; performance and Windows hardware
compatibility still need real-machine testing. GPU acceleration is not
validated in this CPU matrix.

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
python scripts/check_licenses.py
cargo test --locked --release
# Download once, then run actual inference without network access:
cargo run --locked --release -- download --cache-dir models
VALLE_ASR_TEST_CACHE=models cargo test --locked --release --test real_models -- --ignored --nocapture
```

See [THIRD_PARTY.md](https://github.com/openvalle/valle-asr/blob/main/THIRD_PARTY.md) for exact source revisions, affected code,
fixture provenance, preserved notices and dependency licenses. Only
permissively licensed references are used. Project license: Apache-2.0.

## Crate development and release checks

The crate targets Rust 1.99+ and edition 2024. Run formatting, Clippy, unit/API
tests, documentation and packaging checks before a release. Public APIs must
document units, callback execution, resource ownership and failure conditions;
missing public documentation and broken Rustdoc links fail validation. The README
provides the crate-level documentation, and its Rust example is compiled as a
documentation test.
Changes to native sources must retain upstream notices, source hashes and
reproducible patches. The permissive-license policy in THIRD_PARTY.md applies.

The default features are `qwen3` and `cli`. Library consumers use
`default-features = false` with `features = ["qwen3"]` and local model
directories. No-default-features builds expose the model-independent core and
the model catalog. The `cli` feature enables the backend; Clap, hound, tempfile,
ureq and ring are excluded from builds that omit `cli`. This ASR backend uses portable Rust CPU inference.

Each independent Windows/Linux/macOS workflow checks supported feature
combinations, warning-free API documentation and `cargo publish --dry-run`,
which extracts and builds the exact crate archive without uploading it.
Packaging explicitly includes source, the model catalog and third party
notices. Tests and their audio fixtures stay in the repository, where CI runs
them. Model weights, local artifacts and build caches are excluded.

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo test --locked --no-default-features
cargo doc --locked --no-default-features --features qwen3 --no-deps --lib
cargo package --locked --list
cargo publish --locked --dry-run
```

The current release is version 0.3.0. Breaking public API changes require a minor
version increment before 1.0; compatible fixes use a patch increment. Document
changes in the release notes and tag each published version. The minimum Rust
version is declared in `Cargo.toml` and matches the toolchain used in CI.

Before publishing, require all six platform/architecture workflows, including
the explicit real-model tests, to pass, review the archive's contents and licenses, and confirm
the version is new on crates.io. A dry run does not reserve the crate name or
verify the publisher account's ownership. Publishing a GitHub Release tagged `v<version>` triggers
`.github/workflows/release.yml`. The tag must match `Cargo.toml` and belong to
main history; all six independent platform/architecture workflows must have
passed at that exact commit. The workflow uses the repository secret
`CARGO_REGISTRY_TOKEN`, validates the archive before uploading, and checks its
SHA-256 against crates.io. Re-running an identical published archive is safe;
an existing version with different contents is rejected. Manual dispatch can
validate an existing tag with `publish: false` before uploading it.

## 0.3.0 release notes

- The library takes mono 16 kHz f32 samples through `AudioSource`, a slice or
  any source that reads from an offset. `AsrEngine::transcribe` takes a slice;
  `transcribe_source` streams segments from any source to a sink. Reading WAV
  and writing JSON moved into the CLI, and `Audio`, `WavChunks`, `AudioChunk`
  and `JsonTranscriptWriter` left the public API.
- `AsrModel` has a single streaming `transcribe` method.
- In-memory samples are cut at pauses near the chunk limit, as files were.
- Errors are the `valle_asr::Error` enum; `Cancelled` is one of its variants.
  anyhow is no longer a dependency.
- The library no longer downloads models: the `download` feature and the
  `cache` module are gone, and `valle_asr::models` keeps the catalog as data
  with `ModelSpec::missing` to check a directory offline. The CLI keeps a
  downloader for development and CI, now on ureq and ring.
- The mel spectrum is a matrix product. A library build with `qwen3` on macOS
  arm64 depends on 137 crates, and the default build with the CLI on 165,
  down from 225.
- Attention shares key/value heads across query groups without copying them.
- The output of the CLI is unchanged.

## 0.2.0 release notes

- Native CPU support for Windows, Linux and macOS on both x86_64 and ARM64.
- Real Qwen3-ASR and ForcedAligner 0.6B validation on all six targets, including
  English/Chinese recognition, word timestamps, long files and cancellation.
- Six independent native CI workflows with architecture-specific build caches
  and shared revision-pinned model weights.
- Release and permissive-license checks cover all six targets; no intentional
  Rust API breaking changes in this release.

## 0.1.0 release notes

- Initial Rust library and CLI with pluggable ASR backends.
- Qwen3-ASR 0.6B/1.7B local loading and Qwen3-ForcedAligner word timestamps.
- Bounded WAV chunking and streamed JSON for long files.
- Revision-pinned, verified model downloads and permissively licensed sources.
- Independent Linux, Windows and macOS CI with real-model and crate checks.
- Shared host cancellation flags, typed cancellation errors and reusable backends.
