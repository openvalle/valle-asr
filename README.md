# valle-asr

Local speech recognition in Rust with selectable model backends and learned
word timestamps. The first backend is **Qwen3-ASR**, paired with
**Qwen3-ForcedAligner**. This repository is independent of `valle`.

[![Linux](https://github.com/openvalle/valle-asr/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/openvalle/valle-asr/actions/workflows/ci-linux.yml)
[![macOS](https://github.com/openvalle/valle-asr/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/openvalle/valle-asr/actions/workflows/ci-macos.yml)
[![Windows](https://github.com/openvalle/valle-asr/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/openvalle/valle-asr/actions/workflows/ci-windows.yml)

## Scope

- Rust **1.99.0**, edition 2024; direct crates use the latest stable releases
  checked on 2026-10-08, with an exact `Cargo.lock`.
- Native CPU inference on Windows, Linux and macOS through Candle. Python,
  libtorch, OpenBLAS and a GPU are not required to transcribe audio.
- Built-in official Qwen3-ASR 0.6B and ForcedAligner 0.6B downloads. Local
  directories can also load the matching original Qwen3-ASR 1.7B architecture;
  1.7B has not yet been included in the validation matrix.
- Mono/stereo integer/float WAV input, converted to mono 16 kHz with low-pass
  resampling. Applications can pass PCM directly. Other media can be decoded
  by the consumer, including Valle's existing FFmpeg integration.
- Long audio is split into bounded chunks. Word timestamps are predicted by a
  separate neural alignment model and offset to the original audio timeline.
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

## Rust API

```rust,no_run
use valle_asr::{AsrEngine, Audio, TranscribeOptions, models::qwen3::Qwen3};

# fn main() -> anyhow::Result<()> {
let mut engine = AsrEngine::new();
engine.register(Qwen3::load(
    "qwen3-asr-0.6b",
    "models/asr",
    Some("models/aligner".into()),
)?)?;
let result = engine.transcribe(
    "qwen3-asr-0.6b",
    &Audio::from_wav("speech.wav")?,
    &TranscribeOptions::default(),
)?;
println!("{}", result.text);
# Ok(())
# }
```

Implement `AsrModel` to register another model family. `AsrEngine` dispatches
to the selected registered instance. Backends receive the same normalized
audio and return the same model-independent result types. Qwen is optional:
`cargo check --no-default-features --lib` builds the core API alone.

## Cache behavior

The model cache defaults to `VALLE_ASR_CACHE` when set, otherwise to the native
user cache directory. `--cache-dir` overrides it. Every revision has its own
directory and cross-process writer lock. Existing files are reused only after
size and SHA-256 validation; interrupted `.part` files are never treated as
models. Downloads are verified before a final rename. Revisions are immutable
while model files are mapped. Do not modify a directory being used for inference.

Tokenizers remain in memory. A bounded resident cache holds one neural model:
all ASR chunks run before the weights are released and all alignment chunks
run. Repeated text-only transcription and standalone alignment reuse their
resident model; full transcription switches models once per stage. This
avoids retaining both sets of F32 weights on small CI runners. `Qwen3::unload`
releases the resident model explicitly.

GitHub Actions caches Cargo dependencies/builds per platform and shares the
revision-pinned model cache across platforms. Model cache keys include the
hash of `models.json`; each restored cache is still validated before use.

## Validation

Linux, macOS and Windows each have an **independent workflow**, triggered in
parallel. Each runs formatting, Clippy, the permissive-license gate, optional
feature compilation, API/unit tests, then actual 0.6B transcription and word
alignment. Missing weights fail the model test.

The real tests check English/Chinese character error rates, non-collapsed
monotonic word spans inside source chunks, and offsets on a two-chunk clip.
Each platform uploads the output JSON and a timing/accuracy receipt. They
validate correctness on hosted CPUs; performance and Windows hardware
compatibility still need real-machine testing. GPU acceleration and Windows
ARM64 are not validated in this initial matrix.

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
python scripts/check_licenses.py
cargo test --locked --release
# Download once, then run actual inference without network access:
cargo run --locked --release -- download --cache-dir models
VALLE_ASR_TEST_CACHE=models cargo test --locked --release --test real_models -- --ignored --nocapture
```

See [THIRD_PARTY.md](THIRD_PARTY.md) for exact source revisions, affected code,
fixture provenance, preserved notices and dependency licenses. Only
permissively licensed references are used. Project license: Apache-2.0.
