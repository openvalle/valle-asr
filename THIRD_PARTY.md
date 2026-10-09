# Third-party provenance and licenses

This document records external implementation references, adapted code, test
fixtures, model weights and dependency licenses. Valle's original code is
Apache-2.0, as declared in the repository [LICENSE](LICENSE). Third-party
material retains its upstream license and notices.

Only permissively licensed projects were used. No GPL, LGPL, AGPL, SSPL,
source-available, or unlicensed project code is included or used as an
implementation reference. In particular, the previously researched
`roj234/qwen3-audio.cpp` is excluded because its own license was not explicit.

This is a new repository and API, not a fork or a wrapper around another ASR
runtime. The Qwen network, model dispatch, cache, audio processing, and CLI are
implemented here in Rust. Mathematical behavior follows the official model.

| Source | Pinned commit | License | Form of use | Affected material |
|---|---|---|---|---|
| [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff) | `7c6daf77a2421100f5fb066495372c00129d39ff` | Apache-2.0 | Implementation reference; one translated function | Architecture, prompts, frontend and word segmentation in `src/models/qwen3/`. `text.rs::repair_timestamps` translates `Qwen3ForceAlignProcessor.fix_timestamp` into Rust. The remaining model code is a new Candle implementation. |
| [alan890104/qwen3-asr-rs](https://github.com/alan890104/qwen3-asr-rs/tree/c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc) | `c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc` | MIT | Implementation reference; copied test fixtures | Rust/Candle architecture and tensor-layout reference, with no upstream Rust source files copied. `sample1.wav`, `sample1.txt`, `sample5.wav` and `sample5.txt` are copied into `tests/fixtures/`. |
| [antirez/qwen-asr](https://github.com/antirez/qwen-asr/tree/924694251d9e0f18e5d86bbd06aa3ab5f870002d) | `924694251d9e0f18e5d86bbd06aa3ab5f870002d` | MIT | Implementation reference | CPU inference architecture and preprocessing; no C code copied or linked. |
| [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR/tree/dbc44a65554197539b743f8de37a353a08ce0e8d) | `dbc44a65554197539b743f8de37a353a08ce0e8d` | MIT | Documentation reference | Cross-platform packaging and ASR/aligner pipeline organization; no source code or binaries copied or linked. |

Full upstream licenses and their copyright notices are preserved verbatim:

| Source | Preserved license |
|---|---|
| Qwen3-ASR | [Apache-2.0 notice](third_party/QwenLM__Qwen3-ASR-LICENSE) |
| qwen3-asr-rs | [MIT notice](third_party/alan890104__qwen3-asr-rs-LICENSE) |
| qwen-asr | [MIT notice](third_party/antirez__qwen-asr-LICENSE) |
| CrispASR | [MIT notice](third_party/CrispStrobe__CrispASR-LICENSE) |

## Adapted code and original implementation

The timestamp-repair translation is identified in
`src/models/qwen3/text.rs`. Its upstream Apache-2.0 notice remains in the
license file above. No upstream ASR runtime, C inference source or inference
binary is bundled or linked into this crate.

The CPU weight backend in `src/models/qwen3/weights.rs` is new Valle code using
Candle 0.11.0's public `SimpleBackend`, memmap2 0.9.11's portable mapping API and
safetensors 0.8.0's validated metadata API. Per-tensor mappings are released
after loading each weight. These dependencies use MIT/Apache-2.0 licenses and
were already present transitively. Its direct
BF16-to-F32 conversion follows IEEE bit layout and preserves the quiet-NaN
behavior of Candle's `half` dependency. Regression tests compare all BF16 bit
patterns with Candle's existing loader. Candle and half are MIT/Apache-2.0
dependencies already recorded in `third_party/crates.json`; no source files
from either crate are copied or vendored here.

## Test fixtures

[tests/fixtures/provenance.json](tests/fixtures/provenance.json) records the
upstream repository and commit, file sizes, SHA-256 hashes and reference text
for the four copied English/Chinese fixture files. The upstream MIT notice is
included alongside this provenance in the crate's `third_party/` directory.

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
repositories for external dependencies; workspace packages are excluded from
this third-party report. CI runs `scripts/check_licenses.py` and rejects missing licenses,
GPL-family licenses and any expression without a permitted license choice.
Use of a dual-licensed dependency selects its permissive license option.

## Distribution and updates

`Cargo.toml` includes this document, `third_party/`, and fixture provenance in
the published source archive. Model weights and generated test output are
downloaded or produced separately and are excluded from that archive.

When updating dependencies or borrowing code, recheck the actual license,
record the exact revision and affected files, preserve notices, regenerate the
crate report and run the license check before merging.
