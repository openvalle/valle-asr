# Third-party provenance and licenses

Only permissively licensed projects were used. No GPL, LGPL, AGPL, SSPL,
source-available, or unlicensed project code is included or used as an
implementation reference. In particular, the previously researched
`roj234/qwen3-audio.cpp` is excluded because its own license was not explicit.

This is a new repository and API, not a fork or a wrapper around another ASR
runtime. The Qwen network, model dispatch, cache, audio processing, and CLI are
implemented here in Rust. Mathematical behavior follows the official model.

| Source | Pinned commit | License | Use and affected files |
|---|---|---|---|
| [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff) | `7c6daf77a2421100f5fb066495372c00129d39ff` | Apache-2.0 | Architecture, prompts, frontend and word segmentation reference in `src/models/qwen3/`. `text.rs::repair_timestamps` is a Rust translation of `Qwen3ForceAlignProcessor.fix_timestamp`; the other model code is a new Candle implementation. |
| [alan890104/qwen3-asr-rs](https://github.com/alan890104/qwen3-asr-rs/tree/c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc) | `c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc` | MIT | Rust/Candle architecture and tensor-layout reference; no upstream Rust source files copied. English and Chinese WAV/text fixtures in `tests/fixtures/` are copied from the pinned revision, with exact hashes in `tests/fixtures/provenance.json`. |
| [antirez/qwen-asr](https://github.com/antirez/qwen-asr/tree/924694251d9e0f18e5d86bbd06aa3ab5f870002d) | `924694251d9e0f18e5d86bbd06aa3ab5f870002d` | MIT | CPU inference architecture and preprocessing reference only; no C code copied or linked. |
| [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR/tree/dbc44a65554197539b743f8de37a353a08ce0e8d) | `dbc44a65554197539b743f8de37a353a08ce0e8d` | MIT | Cross-platform packaging and ASR/aligner pipeline documentation reference only; no source code or binaries copied or linked. |

Full upstream license texts are preserved under `third_party/`.

The CPU weight backend in `src/models/qwen3/weights.rs` is new Valle code using
Candle 0.11.0's public `SimpleBackend` and mapped safetensors APIs. Its direct
BF16-to-F32 conversion follows IEEE bit layout and preserves the quiet-NaN
behavior of Candle's `half` dependency. Regression tests compare all BF16 bit
patterns with Candle's existing loader. Candle and half are MIT/Apache-2.0
dependencies already recorded in `third_party/crates.json`; no source files
from either crate are copied or vendored here.

Long-file regression tests generate sparse PCM WAV files in `tests/support/`.
Their real speech inserts repeat the existing MIT-licensed `sample1.wav` at
both ends of a two-hour timeline; no additional audio corpus or external
evaluation code is included. The file decoder, streaming writer and sampling
test helpers are newly implemented in this repository.

## Model weights

`models.json` pins official Qwen model revisions and every file's size and
SHA-256. Both model repositories declare Apache-2.0:

- [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0)
- [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/tree/c7cbfc2048c462b0d63a45797104fc9db3ad62b7)

Weights are downloaded into a separate cache and are not committed to Git.
The official Apache-2.0 text is preserved in
`third_party/QwenLM__Qwen3-ASR-LICENSE`.

## Rust dependencies

`Cargo.lock` records the exact resolved dependency versions. Direct dependencies
were checked against the latest stable crates.io releases on 2026-10-08.
Transitive versions are determined by the latest upstream crates' constraints.
`third_party/crates.json` records crate names, versions, SPDX licenses and source
repositories. CI runs `scripts/check_licenses.py` and rejects missing licenses,
GPL-family licenses and any expression without a permitted license choice.
Use of a dual-licensed dependency selects its permissive license option.

When updating dependencies or borrowing code, recheck the actual license,
record the exact revision and affected files, preserve notices, regenerate the
crate report and run the license check before merging.
