# Native architecture accuracy references

Each NPZ contains a small F32 GGUF (`model`, uint8 bytes) and three complete last-token
logit vectors (`logits`, float32). `gen_arch_accuracy.py` creates the weights;
`architecture_oracle.cpp` evaluates them with the real llama.cpp CPU backend.
OpenVINO does not participate in reference generation.

Reference revisions:

- ggml-org/llama.cpp `476c01efe88aad7880a8132d5d3a415f2ca75139`: all except Muse Glimmer.
- mvafin/llama.cpp `3d677aadec0487430ad59bd3d4a9fc7211721f0f`: Muse Glimmer support.

The cases use distinct nonzero weights and nonuniform norm scales, four query heads,
grouped-query attention (single KV head for Gemma), and token batches `[1,2,3]`, `[4]`,
`[5,6]`. The same inference request preserves state across all three batches.
Gemma2 and Muse Glimmer cross a two-token sliding window; SmolLM3 exercises its fourth,
NoPE layer. ERNIE omits `expert_shared_count`, as the real 21B checkpoint does. Bailing
covers sigmoid routing, biased selection, group filtering, and a shared expert.

Devstral adds three configurations under the existing `llama` and `mistral3` families.
Small models use unequal embedding/query widths. Small 2 crosses reduced original-context
boundaries at positions 2 and 4 to exercise attention temperature; Devstral 2 uses non-default
YaRN correction parameters. Each runs natively and through the loadable Devstral example.

`GGUFArchitectureAccuracy` compiles the native frontend graph on CPU with F32 inference,
F16 KV state and dynamic activation quantization disabled. It checks every logit using
normalized MSE below `1e-5`. Missing references fail the test. These fixtures run in the
regular frontend suite, including offline CI; llama.cpp is only needed to regenerate them.

To regenerate, check out the revision above and build it with `GGML_OPENVINO=OFF`:

```sh
c++ -std=c++17 architecture_oracle.cpp \
    -I "$LLAMA_SRC/include" -I "$LLAMA_SRC/ggml/include" \
    -L "$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
    -lllama -lggml -lggml-base -o architecture_oracle
PYTHONPATH="$LLAMA_SRC/gguf-py" python3 gen_arch_accuracy.py \
    --oracle ./architecture_oracle --architectures llama qwen2 qwen3 phi3 minicpm olmoe \
    hunyuan-dense hunyuan-moe qwen3moe gemma gemma2 exaone4 ernie4_5-moe bailingmoe2 \
    maincoder mistral3 smollm3 mellum deepseek2-ocr devstral-small devstral-small2 devstral2
# Use the Muse-capable revision for --architectures muse-glimmer.
```

For an additional real-checkpoint check, create `<arch>.gguf` in a separate directory,
then run `architecture_oracle <arch>.gguf <arch>.bin "The capital of France is"`.
Set `OV_GGUF_ACCURACY_DATA` to that directory and filter the test to that architecture.
This performs prefill plus twelve reference-token decode steps, requires the same first
prediction and at least 90% matching greedy choices, and records per-step normalized MSE.
It does not claim exact logit agreement for the frontend's lossy quantized-weight path.
Use `OV_GGUF_Q4_K_ZP_F16=1` for Q4_K accuracy comparisons.
