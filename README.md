# DeepSeek-V4-Flash on SM120 (RTX PRO 6000 / RTX 50xx) — vLLM fork + build recipe

Source snapshot and reproducible build recipe for running **DeepSeek-V4-Flash**
(284B MoE, 13B active, 1M context) on **consumer/workstation Blackwell (SM120)**
GPUs — hardware that stock vLLM cannot serve this model on. Verified on
2× RTX PRO 6000 Blackwell 96GB (PCIe, no NVLink), CUDA 13.0, driver 595.58.03.

This tree is [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm)
at branch `lucifer` (commit `7c6bbf4`), plus a `deploy/` directory with the
build/run scripts used on this machine. Upstream vLLM README: [README.vllm.md](README.vllm.md).

## Why stock vLLM doesn't work

vLLM's CUDA implementation of DeepSeek-V4 hard-requires **DeepGEMM** in three
hot paths (block-FP8 linear layers, the grouped o-projection `fp8_einsum`, and
the sparse-attention indexer's MQA-logits kernels). Released DeepGEMM only
ships SM90/SM100 (Hopper / datacenter-Blackwell) kernels — on SM120 every one
of those paths dies with `Unsupported architecture`, plus a cascade of
secondary breakage around the checkpoint's `float8_e8m0fnu` block scales
(CUTLASS asserts fp32 scales; Triton can't bind e8m0 pointers). Upstream
tracking: vllm-project/vllm issues
[#47818](https://github.com/vllm-project/vllm/issues/47818) and
[#48602](https://github.com/vllm-project/vllm/issues/48602).

## The working stack (exact pins — the PR heads are moving targets)

| Component | Source | Pin | What it adds |
|---|---|---|---|
| vLLM | `local-inference-lab/vllm` branch `lucifer` (this tree) | `7c6bbf4` | `SPARSE_MLA_SM120` attention backend, SM120 warmups, FlashInfer-CUTLASS MXFP4 MoE for V4 |
| FlashInfer | `flashinfer-ai/flashinfer` PR [#3395](https://github.com/flashinfer-ai/flashinfer/pull/3395) | `b41aa8dd` | `flashinfer.sparse_mla_sm120`, `swa_indices`, `run_sparse_mla` (no released wheel has these; a later PR revision renames the module — use the pin) |
| DeepGEMM | `deepseek-ai/DeepGEMM` PR [#324](https://github.com/deepseek-ai/DeepGEMM/pull/324) | `aced12c2` | SM120 kernel support; installed standalone (vLLM prefers site-packages `deep_gemm` over its vendored copy) |
| CUTLASS DSL | `nvidia-cutlass-dsl==4.5.2` + `blockscaled_layout.py` from `NVIDIA/cutlass@d80a4e53` | — | `sm120_make_smem_layout_sfa` for FlashInfer's cute-DSL kernels |
| PyTorch | `torch==2.11.0+cu130` | — | matches vLLM build; **never let pip's flashinfer resolver touch torch** |

Recipe credit: [`local-inference-lab/blackwell-llm-docker`](https://github.com/local-inference-lab/blackwell-llm-docker)
(`build-lucifer-cu132.sh`) and Ovidiu Dan's write-up of the `voipmonitor/vllm:lucifer`
image. This repo reproduces that stack in a bare conda env (no Docker), on
CUDA 13.0 instead of 13.2.

## Build

Prereqs: miniconda, CUDA toolkit 13.0+ at `/usr/local/cuda-13.0` (nvcc is
needed at **runtime** too — DeepGEMM/FlashInfer JIT-compile kernels), rustup
(vLLM's Rust frontend), ~90GB RAM for `MAX_JOBS=16`. Full build ≈ 1.5–2h.

```bash
git clone https://github.com/Rawsejet/deepseek-vllm-fork.git ~/vllm-lucifer
~/vllm-lucifer/deploy/build-sm120.sh
```

The script creates a conda env `deepseekv4` and builds/installs everything
with the pins above. See its comments for the sharp edges (torch-clobbering
pip resolver, cubin packages overriding patched kernels, PR drift).

## Run

```bash
hf download deepseek-ai/DeepSeek-V4-Flash --local-dir ~/models/deepseek
deploy/start-deepseek.sh   # interactive: port / context / MTP menus
deploy/stop-deepseek.sh
```

Key launch flags (already in the script): `--attention-backend SPARSE_MLA_SM120`,
`--kernel-config.moe_backend flashinfer_cutlass`, `--kv-cache-dtype fp8
--block-size 256`, TP=2, MTP speculative decoding. First launch spends ~5–6
extra minutes JIT-compiling kernels; later launches reuse the cache.

Sampling per the model card: `temperature=1.0, top_p=1.0`. Thinking mode is
selected per request via `chat_template_kwargs: {"thinking": true}`.

## Measured results (2026-07-17, 2× RTX PRO 6000, TP=2, MTP 2 tokens, 256K ctx)

- **202 tok/s** single-stream decode at short context (thinking mode)
- **161 tok/s** single-stream decode at 4.5K-token context, **TTFT 0.58s**
- Needle-in-haystack retrieval correct at 4–5K context (sparse indexer verified)
- Matches the ~210 tok/s reference from the same stack in Docker.

## Operational notes (learned the hard way)

- First launch JIT-compiles kernels. **Cap compile parallelism** — the launch
  script sets `MAX_JOBS=4 NVCC_THREADS=1` and `VLLM_DEEP_GEMM_WARMUP=skip`;
  without these, 32 parallel ~6GB nvcc/cicc instances OOM a 91GB machine.
  Compiled kernels cache to disk; later launches take ~2 min.
- Runtime JIT needs the **CUDA 13.2** toolchain (conda env-local): nvcc 13.0
  fails template resolution on the SM120 fp8_fp4 kernels. The launch script
  sets `CUDA_HOME=$CONDA_PREFIX` plus `CPATH`/`LIBRARY_PATH` for conda's
  header/stub layout.
- If you ever re-checkout DeepGEMM to a different commit, `rm -rf build`
  before reinstalling — setuptools silently reuses stale compiled objects
  and you get baffling template-arity mismatches at JIT time.
