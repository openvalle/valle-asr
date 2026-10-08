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
- Complete WAV files are decoded and transcribed in bounded chunks, with
  silence-aware boundaries. Audio and transcript RAM do not grow with duration
  when using the file API/CLI. Word timestamps retain the original timeline.
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
command. There is no ASR file-duration limit. The decoder reads at most the
current chunk plus resampling neighbors, mixes channels during reading, and
preserves the global resampling phase across boundaries. A 30-second normalized
16 kHz chunk contains 1,920,000 bytes of F32 PCM, regardless of file duration;
source-rate buffers, model weights and inference tensors need additional RAM.
Input sample rates up to 384 kHz are supported by the file decoder.

Within the last three seconds of a chunk (or the last quarter of a small chunk),
the decoder prefers a quiet interval of at least 120 ms. It does not drop or
duplicate source samples. Continuous speech without a suitable pause still
uses the hard chunk limit; this is not a full VAD or overlap/deduplication system.
Exact digital-zero padding is omitted from neural inference while preserving
its offset. Other low-volume sounds are not classified as silence.

With word timestamps, ASR runs first and writes interval/text records to a
temporary file. Its neural weights are released before the alignment pass
re-reads the same WAV. Each neural model loads once per stage. Final segments
are emitted during alignment; text-only mode emits them during the first pass.
The JSON writer also spools text on disk so that the existing full `text` field
does not require a growing RAM buffer. Temporary disk usage and processing time
grow with duration, and temporary files are removed on success or error.
`--output` replaces the destination only after successful completion.

The current decoder supports standard RIFF WAV, whose 32-bit size fields impose
an approximately 4 GiB container limit (about 37 hours for 16 kHz mono PCM16).
RF64 and compressed-media file decoding are not implemented here. Consumers
can use their own media decoding for other formats.

## Rust API

```rust,no_run
use std::{fs::File, io::BufWriter};
use valle_asr::{AsrEngine, JsonTranscriptWriter, TranscribeOptions, models::qwen3::Qwen3};

# fn main() -> anyhow::Result<()> {
let mut engine = AsrEngine::new();
engine.register(Qwen3::load(
    "qwen3-asr-0.6b",
    "models/asr",
    Some("models/aligner".into()),
)?)?;
let mut output = JsonTranscriptWriter::new(BufWriter::new(File::create("transcript.json")?))?;
let summary = engine.transcribe_file(
    "qwen3-asr-0.6b",
    "speech.wav",
    &TranscribeOptions::default(),
    &mut |segment| output.write_segment(segment),
)?;
output.finish(&summary)?;
# Ok(())
# }
```

The existing `transcribe(&Audio, ...)` API remains available for already-loaded
PCM and intentionally returns an in-memory `Transcript`. For long files, use
`transcribe_file` and a sink that persists or displays segments without retaining
them. A consumer collecting every callback into a vector will still grow RAM.

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
Text files use LF on every platform so Windows checkout does not change
content-based cache keys; WAV fixtures remain binary.

## Validation

Linux, macOS and Windows each have an **independent workflow**, triggered in
parallel. Each runs formatting, Clippy, the permissive-license gate, optional
feature compilation, API/unit tests, then actual 0.6B transcription and word
alignment. Missing weights fail the model test.

The real tests check English/Chinese character error rates, non-collapsed
monotonic word spans inside source chunks, and offsets on a two-chunk clip.
Every platform also runs a two-hour sparse WAV through the streaming API/JSON
writer, continuous resampling tests, and real Qwen ASR/alignment on speech at
both ends of a two-hour sparse WAV. The long fixture is mostly digital silence;
it validates file handling, bounded chunks and global word offsets, not two
hours of continuous-speech accuracy or throughput.
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
