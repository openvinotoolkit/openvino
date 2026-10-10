# Bonsai 2 27B on the OpenVINO GPU plugin with the TernOCL int2 kernels — step by step

From an empty directory to Bonsai 2 27B decoding on an Intel Xe2 GPU (discrete
or integrated). How the int2 `FullyConnected` path works is described in
[ternocl_int2_architecture.md](ternocl_int2_architecture.md).

---

## 0. What you need

| | |
|---|---|
| GPU | Intel Xe2, discrete (Arc Pro B-series) or integrated (Lunar Lake Arc 140V) |
| GPU runtime | Intel compute runtime with OpenCL — the int2 path runs on the plugin's **OpenCL** runtime |
| Compiler | Intel oneAPI (`icx`/`icpx`) |
| Tools | CMake ≥ 3.16, Ninja, git, Python 3.10+ |
| Disk | ~150 GB: OpenVINO build ~25 GB, Bonsai 1 27B checkpoint 51 GB + fp16 IR 51 GB (template, see step 3), Bonsai 2 GGUF 7.2 GB, final IR 7.2 GB |
| RAM | ≥ 64 GB on the machine that runs the exports (step 3 loads the 27B in fp16) |

The environment every GPU command needs (put it in a file and `source` it):

```bash
# env.sh
source /opt/intel/oneapi/setvars.sh          # oneAPI compilers and runtime libraries
export WORK=$HOME/ov-bonsai2
export OV_ROOT=$WORK/openvino
export LD_LIBRARY_PATH=$OV_ROOT/bin/intel64/Release:$LD_LIBRARY_PATH
export ZE_AFFINITY_MASK=0                    # multi-GPU hosts: which GPU to use
```

---

## 1. Download the sources

```bash
mkdir -p $WORK && cd $WORK
git clone https://github.com/openvinotoolkit/openvino.git openvino
cd openvino && git submodule update --init --recursive && cd ..      # includes thirdparty/TernOCL

# gguf-py of the PrismML llama.cpp fork: stock gguf does not know the PQ2_0 tensor type
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp llama.cpp-prism
```

The OpenCL kernels come from [TernOCL](https://github.com/libxsmm/TernOCL),
pinned as the submodule `src/plugins/intel_gpu/thirdparty/TernOCL`.

## 2. Build OpenVINO (GPU plugin + TernOCL int2 path) and the benchmark tools

The TernOCL sources are read from the submodule at configure time; set
`-DTERNOCL_ROOT=<dir>` only to build against another TernOCL checkout.

```bash
source env.sh && cd $OV_ROOT
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_FLAGS=-fp-model=precise -DCMAKE_CXX_FLAGS=-fp-model=precise \
  -DGPU_RT_TYPE=OCL \
  -DENABLE_INTEL_CPU=OFF -DENABLE_INTEL_NPU=OFF \
  -DENABLE_PYTHON=OFF -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF \
  -DENABLE_ONEDNN_FOR_GPU=ON -DENABLE_CM_FOR_GPU=ON \
  -DENABLE_SYSTEM_OPENCL=OFF \
  -DTHREADING=TBB_ADAPTIVE
cmake --build build -j $(nproc)
```

`-fp-model=precise` is required with `icx`/`icpx`: their default fast floating-point
model rewrites the `double` division in `ov::Dimension` shape arithmetic as a
multiplication by the reciprocal, which rounds some exact quotients down (e.g.
7450 / 3725 → 1). Reshapes with a `-1` then fail at run time for about one prompt
length in five ("Non-'-1' output dimensions do not evenly divide the input dimensions").

```bash
strings bin/intel64/Release/libopenvino_intel_gpu_plugin.so | grep -c int2_fp16_upcvt_gemm_mt   # non-zero: kernels embedded

cd src/plugins/intel_gpu/tools/int2
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DOpenVINO_DIR=$OV_ROOT/build && cmake --build build
ls build/   # paged_bench_llm  paged_bench_llm_27b  paged_serve_llm_27b
```

The tools link with an RPATH to this build. To compare two OpenVINO builds
with one set of tools, configure them with `-DCMAKE_SKIP_RPATH=ON` and select
the build through `LD_LIBRARY_PATH` (OpenVINO loads the GPU plugin from next
to `libopenvino.so`).

Python side (model preparation only; no GPU, no OpenVINO build needed — it
uses the pip wheel):

```bash
cd $WORK
python3 -m venv venv && ./venv/bin/pip install openvino numpy pyyaml
./venv/bin/pip install "optimum-intel[openvino]==2.1.0" "transformers==5.2.0"   # step 3 only
```

## 3. Get the graph template: Bonsai 27B (1st generation) IR

Bonsai 2 27B ships only as GGUF and has *exactly* the Qwen3.5-27B architecture
of Bonsai 27B. Instead of rebuilding a 54 GB dense checkpoint and re-exporting,
the converter reuses the Bonsai 27B IR as the graph and swaps every weight in
from the GGUF (step 4). So the Bonsai 27B IR is a one-time prerequisite; if you
already have `bonsai27b-u2/` skip to step 4.

```bash
cd $WORK
hf download prism-ml/Ternary-Bonsai-27B-unpacked --local-dir bonsai27b-hf      # 51 GB, 12 safetensors shards

# fp16 export; the 27B is a VLM, so the task is image-text-to-text and it produces several models
./venv/bin/optimum-cli export openvino --model bonsai27b-hf \
    --task image-text-to-text --weight-format fp16 bonsai27b-fp16                # needs ~60 GB RAM

# rewrite the 497 ternary MatMuls as u2 codes + fp16 group scales (exact: the weights already are ternary)
./venv/bin/python $OV_ROOT/src/plugins/intel_gpu/tools/int2/quantize_ir_ternary.py \
    --in  bonsai27b-fp16/openvino_language_model.xml \
    --out bonsai27b-u2/openvino_model.xml
# expected: "rewriting 497 MatMul weights" ... "weights 47.72 GiB -> 6.34 GiB"
```

Only `bonsai27b-u2/openvino_model.xml` (+`.bin`) is used further; the optional
`--verify` self-check of step 4 also reads `bonsai27b-fp16/openvino_text_embeddings_model.xml`.
The rest of `bonsai27b-fp16` (51 GB) and `bonsai27b-hf` can be deleted.

## 4. Download Bonsai 2 27B and build its IR

```bash
cd $WORK
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir bonsai2-gguf   # 7.2 GB
hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit tokenizer.json tokenizer_config.json chat_template.jinja \
    config.json generation_config.json --local-dir bonsai2-tok         # tokenizer + chat template (for the harness)

./venv/bin/python $OV_ROOT/src/plugins/intel_gpu/tools/int2/bonsai2_gguf_to_ir.py \
    --gguf           bonsai2-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --template-dir   bonsai27b-u2 \
    --gguf-py        llama.cpp-prism/gguf-py \
    --out-dir        bonsai2-27b-u2
```

Needs ~16 GB RAM. Expected log:

```
[ir] ...PQ2_0.gguf: 851 tensors, GDN nv=48 nk=16 hd=128
[ir] hadamard H1024: 401 folded weights, sign widths [5120, 6144, 17408]
[ir] 497 linear MatMuls in template
[ir] weights: 401 ternary, 96 dense, 64 up_proj row folds
[ir] hadamard: 257 rotations inserted, 64 explicit sign multiplies, 129 norm folds
[ir] embedding: ternary 248320x5120, inverse-rotated on the device
[ir] wrote bonsai2-27b-u2
```

Output: `bonsai2-27b-u2/openvino_model.xml` + `.bin` (6.9 GB; u2 weights, fp16
scales, the H_1024 rotation in front of every folded projection) and
`bonsai2-27b-u2/openvino_text_embeddings_model.xml` + `.bin` (0.34 GB: the
ternary table as u8-packed codes + fp16 scales; the GPU gathers the rows,
dequantises them and applies the inverse rotation). How the rotation is executed
is described in section 7 of the architecture doc. Self-check of the mapping
(optional): run the tool on the Bonsai 1 GGUF with `--verify --template-embed
bonsai27b-fp16/openvino_text_embeddings_model.xml`; it must print `verify OK`
(every ternary matrix bit-identical to the template).

## 5. Run

### 5.1 Single-sequence decode (256 tokens, greedy)

```bash
source env.sh
TOOLS=$OV_ROOT/src/plugins/intel_gpu/tools/int2
IDS=$(cat $TOOLS/prompt_photosynthesis_27b.txt)      # chat-templated "Tell me about photosynthesis in 200 words"

BENCH_PRECISION=f16 BENCH_MAX_LEN=512 BENCH_NO_EOS=1 \
  $TOOLS/build/paged_bench_llm_27b $WORK/bonsai2-27b-u2/openvino_model.xml \
  $WORK/bonsai2-27b-u2/openvino_text_embeddings_model.xml GPU 256 "$IDS"
```

The same command runs on discrete and integrated Xe2. The first run builds the
OpenCL programs (the driver caches the binaries for later processes), so time
the second run. The bench prints TTFT, decode throughput and the generated ids:

```
generated_ids  =760,1156,6587,264,61446,15673,314,7022,71163,303,6681,466,12805,220,17,15,15,4105,13,...
```

which decodes (Bonsai 2 tokenizer) to *"The user wants a concise explanation of
photosynthesis in exactly or approximately 200 words. Let me craft a clear,
informative paragraph..."* followed by the essay. Runs are deterministic.

### 5.2 Accuracy with a standard harness

```bash
./venv/bin/pip install "lm_eval>=0.4.13" transformers
cd $WORK && mkdir -p eval && cd eval
$WORK/venv/bin/python $TOOLS/lm_eval_ov.py \
  --lm $WORK/bonsai2-27b-u2/openvino_model.xml --embed $WORK/bonsai2-27b-u2/openvino_text_embeddings_model.xml \
  --tokenizer $WORK/bonsai2-tok --serve $TOOLS/build/paged_serve_llm_27b \
  --tasks gsm8k_cot_llama --batch 16 --think medium --out .          # add --limit 100 for a quick check
```

`$TOOLS/run_lm_eval_ov.sh <model-dir> <tokenizer-dir> <out-dir>` wraps the same
call (`PY`, `LIMIT`, `BATCH`, `THINK`, `TASKS`, `SERVE`, `GPUS` from the environment).
On a multi-GPU host, `--gpus 0,1,...` (`GPUS=...`) splits the requests over one
serving process per GPU; `--batch` is then per GPU.
Full test set (1319 examples, thinking, up to 4096 generated tokens):
`exact_match 0.970` (1279/1319). First 100: ~0.96-0.98 depending on the slice.

### 5.3 Other ternary checkpoints (Bonsai 1.7B-8B, CAT-Q)

The same impl serves any ternary checkpoint; nothing above is specific to the
27B except the Hadamard rotation. For a Qwen3-family checkpoint, export the fp16
IR and rewrite its weights:

```bash
./venv/bin/optimum-cli export openvino --model <hf-checkpoint> \
    --task text-generation-with-past --weight-format fp16 <name>-fp16
./venv/bin/python $TOOLS/quantize_ir_ternary.py --in <name>-fp16/openvino_model.xml \
    --out <name>-u2/openvino_model.xml          # CAT-Q: add --skip-n 151936 (lm_head stays 16-bit)
```

`bench_ternary_llms.sh` runs batch-1 greedy decode on a list of IRs (Qwen3 ones
on `paged_bench_llm`, 27B-class ones given as `<dir>:<embeddings.xml>` on
`paged_bench_llm_27b`) and prints TTFT, decode tok/s and 256 tokens /
(prefill + generation), averaged over reps 2..`REPS`:

```bash
$TOOLS/bench_ternary_llms.sh $WORK/bonsai17b-u2 $WORK/bonsai8b-u2 \
    $WORK/bonsai27b-u2:$WORK/bonsai27b-fp16/openvino_text_embeddings_model.xml
```

Arc Pro B70 (end-to-end tok/s): Bonsai 1.7B 246, 4B 189, 8B 160, 27B 52.7,
CAT-Q 1.7B 198, 8B 124, 32B 44.6. The small models are bound by host-side
overhead, so the host CPU moves them by up to 1.6x; compare builds on one machine.

## 6. Knobs that matter

On/off variables take `1` or `0` (also `true`/`false`, `on`/`off`); any other
value is rejected with an error.

| Variable | Default | Effect |
|---|---|---|
| `OV_TERNOCL_INT2_FUSE_HADAMARD` | 1 | fuse the Hadamard rotation into the FC; `0` leaves it as separate graph ops (debug fallback, slower) |
| `OV_TERNOCL_INT2_INT8_PREFILL` | 0 | `1`: prompts and batches (M > 8) on the int2 x int8 DPAS kernel (activations quantized to int8 per 128-group); same weights, GSM8K within the standard error of the default |
| `OV_TERNOCL_INT2_DISABLE` | 0 | `1`: use the stock OpenVINO FC kernels instead of TernOCL (the Hadamard rotation then runs as graph ops and the gate/up merge follows the stock heuristic) |
| `OV_TERNOCL_INT2_DEBUG` | 0 | `1`: which FCs the TernOCL impl accepted and why others were rejected |
| `OV_TERNOCL_INT2_CFG_DEBUG` | 0 | `1`: every OpenCL program built and the tile chosen per (shape, M class) |
| `OV_TERNOCL_HADAMARD_DEBUG` | 0 | `1`: trace the rotation fusion per FC (expect "fused 257 input rotations") |
| `BENCH_MAX_LEN` | 512 | context the bench reserves; prompt + new tokens must fit |

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Cannot load library ... libsvml.so` | oneAPI not sourced in the process that runs the binary (`env.sh`) |
| `TernOCL kernel ... not found` at configure | the submodule is not checked out: `git submodule update --init src/plugins/intel_gpu/thirdparty/TernOCL` (or pass `-DTERNOCL_ROOT=<dir>`) |
| `ternocl int2: kernel build failed (...)` | the OpenCL compiler rejected the TernOCL source; the build log follows the message |
| `... carries a Hadamard input transform but the TernOCL impl rejected it` | an FC of the rotated model fell outside the impl's rules; the reason is in the message |
| `ModuleNotFoundError: yaml` from the converter | `pip install pyyaml` (gguf-py of the fork imports it) |
| converter: `need the PrismML llama.cpp fork's gguf-py` | pass `--gguf-py <fork>/gguf-py`; stock gguf lacks type id 142 (PQ2_0) |
| fluent nonsense output | wrong tokenizer for decoding (Bonsai 2 uses the Qwen3.5 vocabulary) — or a converter/mapping error: run `--verify` |
