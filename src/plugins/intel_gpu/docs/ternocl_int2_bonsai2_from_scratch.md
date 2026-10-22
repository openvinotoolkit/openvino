# Bonsai 2 27B on the OpenVINO GPU plugin with the TernOCL int2 kernels — step by step

From an empty directory to Bonsai 2 27B decoding on an Intel Xe2 GPU (discrete
or integrated) through OpenVINO GenAI, with performance from `llm_bench` and
accuracy from `who_what_bench`. How the int2 `FullyConnected` path works is described in
[ternocl_int2_architecture.md](ternocl_int2_architecture.md).

---

## 0. What you need

| | |
|---|---|
| GPU | Intel Xe2, discrete (Arc Pro B-series) or integrated (Lunar Lake Arc 140V) |
| GPU runtime | Intel compute runtime with OpenCL — the int2 path runs on the plugin's **OpenCL** runtime |
| Compiler | GCC (tested with 14.3); Intel oneAPI `icx`/`icpx` also works, see the note in step 2 |
| Tools | CMake ≥ 3.26, Ninja, git, Python 3.11+ |
| Disk | ~160 GB: OpenVINO + GenAI build ~35 GB, Bonsai 1 27B checkpoint 51 GB + fp16 IR 51 GB (template, see step 3), Bonsai 2 GGUF 7.2 GB, final IR 7.2 GB |
| RAM | ≥ 64 GB on the machine that runs the exports (step 3 loads the 27B in fp16) |

The environment every command needs (put it in a file and `source` it):

```bash
# env.sh
export WORK=$HOME/ov-bonsai2
export OV_ROOT=$WORK/openvino
export TOOLS=$OV_ROOT/src/plugins/intel_gpu/tools/int2
export ZE_AFFINITY_MASK=0                    # multi-GPU hosts: which GPU to use
```

and, for the GPU runs of step 5, the OpenVINO + GenAI build of step 2:

```bash
# genai.sh
source $WORK/env.sh
source $OV_ROOT/install/setupvars.sh
source $WORK/bench-venv/bin/activate
```

---

## 1. Download the sources

```bash
mkdir -p $WORK && cd $WORK
git clone https://github.com/openvinotoolkit/openvino.git openvino
cd openvino && git submodule update --init --recursive && cd ..      # includes thirdparty/TernOCL
git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git openvino.genai   # includes openvino_tokenizers

# gguf-py of the PrismML llama.cpp fork: stock gguf does not know the PQ2_0 tensor type
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp llama.cpp-prism
```

The OpenCL kernels come from [TernOCL](https://github.com/libxsmm/TernOCL),
pinned as the submodule `src/plugins/intel_gpu/thirdparty/TernOCL`.

## 2. Build OpenVINO with GenAI, and the benchmark environment

OpenVINO is built with GenAI and OpenVINO Tokenizers as extra modules, so the
`openvino`, `openvino_genai` and tokenizer libraries come from one build. The
TernOCL sources are read from the submodule at configure time; set
`-DTERNOCL_ROOT=<dir>` only to build against another TernOCL checkout.

The Python environment for `llm_bench` and `who_what_bench` comes first, since
the build compiles the Python bindings for it:

```bash
source env.sh && cd $WORK
python3 -m venv bench-venv
for t in llm_bench who_what_benchmark; do
  bench-venv/bin/pip install -r openvino.genai/tools/$t/requirements.txt
done
bench-venv/bin/pip install --no-deps -e openvino.genai/tools/who_what_benchmark
bench-venv/bin/pip uninstall -y openvino openvino-tokenizers openvino-genai   # use this build's packages instead
```

```bash
cd $OV_ROOT
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$OV_ROOT/install \
  -DOPENVINO_EXTRA_MODULES=$WORK/openvino.genai \
  -DENABLE_PYTHON=ON -DPython3_EXECUTABLE=$WORK/bench-venv/bin/python \
  -DGPU_RT_TYPE=OCL -DENABLE_SYSTEM_OPENCL=OFF \
  -DENABLE_INTEL_CPU=ON -DENABLE_INTEL_NPU=OFF \
  -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF \
  -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=OFF \
  -DENABLE_OV_TF_FRONTEND=OFF -DENABLE_OV_TF_LITE_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF
cmake --build build -j $(nproc) && cmake --install build
```

The CPU plugin runs the tokenizers. `ENABLE_SYSTEM_OPENCL=OFF` keeps GenAI's
OpenCL code on the bundled headers; mixing them with a newer system
`CL/opencl.hpp` fails to compile.

With `icx`/`icpx` instead of GCC, also pass `-DCMAKE_C_FLAGS=-fp-model=precise
-DCMAKE_CXX_FLAGS=-fp-model=precise`: their default fast floating-point model
rewrites the `double` division in `ov::Dimension` shape arithmetic as a
multiplication by the reciprocal, which rounds some exact quotients down (e.g.
49 / 49 → 0). Reshapes with a `-1` then fail at run time for about one prompt
length in five ("Non-'-1' output dimensions do not evenly divide the input dimensions").
Also pass `-DENABLE_XGRAMMAR=OFF` (GenAI's xgrammar dependency does not link with
`icpx`; it only serves structured output), and source the oneAPI environment
before running, for the compiler runtime libraries. On an Arc Pro B70 the icx
build generates the same tokens as GCC, with a ~10% shorter first token and the
same decode speed.

```bash
source $WORK/genai.sh
python -c "import openvino as ov, openvino_genai; print(ov.get_version(), ov.Core().available_devices)"   # ... ['CPU', 'GPU']
```

Python side of the model preparation (steps 3-4; no GPU, no OpenVINO build
needed — it uses the pip wheels in a separate venv):

```bash
cd $WORK
python3 -m venv venv && ./venv/bin/pip install openvino openvino-tokenizers numpy pyyaml
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
./venv/bin/hf download prism-ml/Ternary-Bonsai-27B-unpacked --local-dir bonsai27b-hf      # 51 GB, 12 safetensors shards

# fp16 export; the 27B is a VLM, so the task is image-text-to-text and it produces several models
./venv/bin/optimum-cli export openvino --model bonsai27b-hf \
    --task image-text-to-text --weight-format fp16 bonsai27b-fp16                # needs ~60 GB RAM

# rewrite the 497 ternary MatMuls as u2 codes + fp16 group scales (exact: the weights already are ternary)
./venv/bin/python $TOOLS/quantize_ir_ternary.py \
    --in  bonsai27b-fp16/openvino_language_model.xml \
    --out bonsai27b-u2/openvino_model.xml
# expected: "rewriting 497 MatMul weights" ... "weights 47.72 GiB -> 6.34 GiB"
```

The converter of step 4 uses `bonsai27b-u2/openvino_model.xml` (+`.bin`) as its
graph; the GenAI model directory of step 4 takes the tokenizer, config and
vision encoders of `bonsai27b-fp16` and the processor configs of `bonsai27b-hf`.
Copy those (step 4) before deleting the fp16 language model
(`bonsai27b-fp16/openvino_language_model.*`, 51 GB) and the `bonsai27b-hf` shards.

## 4. Download Bonsai 2 27B and build its IR

```bash
cd $WORK
./venv/bin/hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir bonsai2-gguf   # 7.2 GB

./venv/bin/python $TOOLS/bonsai2_gguf_to_ir.py \
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

For GenAI, Bonsai 2 27B is a Qwen3.5 vision-language model (`VLMPipeline`):
its model directory holds the two Bonsai 2 models next to the Qwen3.5 tokenizer,
config and vision encoders of the template export (Bonsai 2 uses the same
vocabulary; the vision encoders are loaded but text-only prompts never run them):

```bash
mkdir -p bonsai2-27b-genai && cd bonsai2-27b-genai
ln -s ../bonsai2-27b-u2/openvino_model.xml openvino_language_model.xml
ln -s ../bonsai2-27b-u2/openvino_model.bin openvino_language_model.bin
ln -s ../bonsai2-27b-u2/openvino_text_embeddings_model.xml ../bonsai2-27b-u2/openvino_text_embeddings_model.bin .
ln -s ../bonsai27b-fp16/*.json ../bonsai27b-fp16/chat_template.jinja \
      ../bonsai27b-fp16/openvino_*tokenizer.* ../bonsai27b-fp16/openvino_vision_embeddings_* .
cp ../bonsai27b-hf/*processor_config.json .     # the processor configs WWB loads for a VLM
cd ..
```

## 5. Run

```bash
source $WORK/genai.sh
```

### 5.1 Performance (`llm_bench`)

```bash
cd $WORK/openvino.genai/tools/llm_bench
python benchmark.py -m $WORK/bonsai2-27b-genai -d GPU \
    -p "Explain photosynthesis in two sentences." --disable_prompt_permutation -ic 256 -n 3
```

The warm-up iteration builds the OpenCL programs for the new shapes; the
average is over the following iterations. Arc Pro B70 (GCC build): 1st token
~100 ms, 2nd token 19.3 ms (52 tokens/s). The same command runs on integrated
Xe2. Batch-1 decode is partly host-bound, so the host CPU moves these numbers;
compare builds on one machine.

### 5.2 Accuracy (`who_what_bench`)

There is no fp16 Bonsai 2, so the reference is the same model on the stock
OpenVINO FC kernels; the score then measures what the TernOCL kernels change:

```bash
cd $WORK/openvino.genai/tools/who_what_benchmark
OV_GPU_ENABLE_TERNOCL_INT2=0 wwb --base-model $WORK/bonsai2-27b-genai --gt-data gt_stock.csv \
    --model-type visual-text-only --genai --device GPU          # reference; the stock kernels run at ~2 tokens/s
wwb --target-model $WORK/bonsai2-27b-genai --gt-data gt_stock.csv \
    --model-type visual-text-only --genai --device GPU
```

### 5.3 Other ternary checkpoints (Bonsai 1.7B-8B, CAT-Q)

The same impl serves any ternary checkpoint; nothing above is specific to the
27B except the Hadamard rotation. A Qwen3-family checkpoint is a plain LLM for
GenAI (`LLMPipeline`). Export the fp16 IR, rewrite its weights and copy the
tokenizer and config files next to it:

```bash
cd $WORK
./venv/bin/optimum-cli export openvino --model <hf-checkpoint> \
    --task text-generation-with-past --weight-format fp16 <name>-fp16
./venv/bin/python $TOOLS/quantize_ir_ternary.py --in <name>-fp16/openvino_model.xml \
    --out <name>-u2/openvino_model.xml          # CAT-Q: add --skip-n 151936 (lm_head stays 16-bit)
cp <name>-fp16/*.json <name>-fp16/openvino_*tokenizer.* <name>-u2/
```

Then `llm_bench` as in 5.1, and `wwb --model-type text` against `<name>-fp16`
(or against the stock kernels as in 5.2). Bonsai 8B on an Arc Pro B70: 168
tokens/s; WWB similarity 0.988 between TernOCL and the stock kernels, 0.77
between the ternary and the fp16 model (the stock kernels score the same 0.77).

## 6. Knobs that matter

GPU plugin options (`options.inl`); each can be passed as a compile property or
set through the environment as `OV_<NAME>`, e.g. `OV_GPU_TERNOCL_INT2_INT8_PREFILL=1`.
On/off values are `1`/`0`, `true`/`false`, `on`/`off` or `yes`/`no`.

| Option | Default | Effect |
|---|---|---|
| `GPU_TERNOCL_INT2_INT8_PREFILL` | 0 | `1`: prompts and batches (M > 8) on the int2 x int8 DPAS kernel (activations quantized to int8 per 128-group); same weights, GSM8K within the standard error of the default |
| `GPU_ENABLE_TERNOCL_INT2` | 1 | `0`: use the stock OpenVINO FC kernels instead of TernOCL (the Hadamard rotation then runs as graph ops and the gate/up merge follows the stock heuristic) |
| `GPU_ENABLE_HADAMARD_FC_FUSION` | 1 | debug-caps builds only; `0` leaves the Hadamard rotation as separate graph ops (slower) |
| `GPU_ENABLE_RMS_ROPE_FUSION` | 1 | `0` keeps the per-head Q/K RMSNorm as its own kernel instead of folding it into RoPE. No effect on Bonsai 2 27B, whose heads are only partly rotated |

With `-DENABLE_DEBUG_CAPS=ON`, `OV_VERBOSE=3` logs which FCs the TernOCL impl
accepts and how many rotations were fused ("fused 257 input rotations"), and
`OV_VERBOSE=4` adds every rejection reason and every OpenCL program it builds.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Segmentation fault` when Python exits, after all results are printed | a teardown race outside this path: it also occurs with upstream OpenVINO master and with TernOCL disabled; the results are complete |
| GenAI build: `CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KHR` was not declared | system OpenCL C++ headers mixed with the bundled ones: configure with `-DENABLE_SYSTEM_OPENCL=OFF` |
| `Check '!chat_template.empty()' failed` (`llm_bench --apply_chat_template`) | the checkpoint has no chat template; drop the option or add `chat_template.jinja` before converting the tokenizer |
| `TernOCL kernel ... not found` at configure | the submodule is not checked out: `git submodule update --init src/plugins/intel_gpu/thirdparty/TernOCL` (or pass `-DTERNOCL_ROOT=<dir>`) |
| `ternocl int2: kernel build failed (...)` | the OpenCL compiler rejected the TernOCL source; the build log follows the message |
| `... carries a Hadamard input transform but the TernOCL impl rejected it` | an FC of the rotated model fell outside the impl's rules; the reason is in the message |
| `ModuleNotFoundError: yaml` from the converter | `pip install pyyaml` (gguf-py of the fork imports it) |
| converter: `need the PrismML llama.cpp fork's gguf-py` | pass `--gguf-py <fork>/gguf-py`; stock gguf lacks type id 142 (PQ2_0) |
| fluent nonsense output | wrong tokenizer for decoding (Bonsai 2 uses the Qwen3.5 vocabulary) — or a converter/mapping error: run `--verify` |
