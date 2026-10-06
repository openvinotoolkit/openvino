# Accuracy debugging

Identify the failing execution path before choosing tools:

- Native decoder: use [architecture accuracy references](../../../../src/frontends/gguf/tests/test_data/arch_accuracy/README.md)
  and compare full logits on identical token histories through prefill and cached decode.
  Isolate builder/converter behavior from stateful and GenAI adaptation when localizing a mismatch.
- Native mmproj: use [encoder accuracy references](../../../../src/frontends/gguf/tests/test_data/mmproj_accuracy/README.md).
  Compare the same preprocessed inputs and positional indices, then raw versus adapted embeddings.
  Check each declared modality and relevant dynamic sizes.
- llama.cpp OpenVINO backend: use the relevant bisection and backend-debug sections of
  [debugging_accuracy.md](../../../../src/frontends/gguf/docs/debugging_accuracy.md).
  `GGML_OPENVINO_*` controls belong to that backend; they do not configure native frontend runs.

Use real llama.cpp/ggml CPU outputs as the model and layout-sensitive operation reference.
Record the reference revision, checkpoint, input history, and precision/cache settings.
Keep comparisons against quantized execution and against F32 arithmetic on represented weights
separate: the same checkpoint can still use different activation quantization or weight
requantization. Follow the applicable acceptance limits in
[supported models](../../../../src/frontends/gguf/docs/supported_models.md), preserving failed results.

Localize with the cheapest relevant comparison: prefill versus decode, raw versus adapted graph,
or a suspect encoder branch. Compare elementwise values and error metrics; summary statistics
cannot detect permutations, and a cosine cliff alone does not prove one. Consult the debugging
guide's bug archetypes when layout, RoPE, VIEW strides, or state updates are implicated.

Once localized, add a focused regression at dimensions that expose the failure, using a real
ggml oracle. Check all affected conversion paths. After a shared-path fix, rerun the supported
architecture/projector regressions that exercise it, including classification and numerical output.
