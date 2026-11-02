# Activations Scale Factor from Model rt_info: Behavior Change in OpenVINO 2026.5

*By Jade Cho | November 2026*

Some models overflow FP16 when they run on Intel GPU. The GPU plugin can fix this with **activations scaling**: it scales activations down by a factor, runs the layers in FP16, and scales the result back up. The factor is set with the `ACTIVATIONS_SCALE_FACTOR` (ASF) property. A model can also carry the factor in its IR.

Up to 2026.4, the GPU plugin ignored the ASF in the IR of any LLM. OpenVINO 2026.5 changes this: the plugin applies the IR's ASF to an LLM when the value is not `8.0`. This post explains the old behavior and the change in 2026.5.

---

## What Is ASF?

`ov::hint::activations_scale_factor` (`ACTIVATIONS_SCALE_FACTOR`) is a float.

| Value | Effect |
|---|---|
| `<= 0` (default `-1.0`) | Activations scaling is off |
| `> 0` | Activations scaling is on with this factor |

Activations scaling works only when the inference precision is FP16. When it is on, the plugin inserts scale-down `Multiply` ops before layers such as MatMul and Convolution, and moves the matching scale-up ops down the graph so that they can be merged with other ops. This keeps large intermediate values within the FP16 range.

Activations scaling is not free. It adds ops to the graph and can make a model slower. For some models it also changes the output. So it should be on only for models that need it.

### Where the Value Comes From

The plugin reads the value from these sources. A source higher in the list wins.

| Priority | Source | Example |
|---|---|---|
| 1 | Environment variable | `OV_ACTIVATIONS_SCALE_FACTOR=4` |
| 2 | User property | `core.compile_model(model, "GPU", {"ACTIVATIONS_SCALE_FACTOR": 4.0})` |
| 3 | Model rt_info (IR) | `<rt_info><runtime_options><ACTIVATIONS_SCALE_FACTOR value="4.0" /></runtime_options></rt_info>` |
| 4 | Plugin default | `-1.0` (off) |

This post is only about priority 3: when the plugin uses the value in the IR. A value set by the user (priority 1 or 2) is always used, before and after this change.

You can check the value that the plugin used:

```python
compiled = core.compile_model(model, "GPU")
print(compiled.get_property("ACTIVATIONS_SCALE_FACTOR"))  # -1.0 means off
```

---

## Why the IR's ASF Was Ignored for LLMs

Optimum Intel writes `ACTIVATIONS_SCALE_FACTOR=8.0` into the IR of every text-generation model, every VLM language model, and every stateful sub-model (`_set_runtime_options` in `optimum/exporters/openvino/convert.py`). It does not check whether the model needs scaling.

In early 2025 the GPU plugin stopped applying this value to LLMs ([#28632](https://github.com/openvinotoolkit/openvino/pull/28632)) because of accuracy and performance issues. Since then, the 8.0 in LLM IRs has had no effect on GPU.

The plugin treats a model as an LLM when it has one of these:

- a KV-cache pattern: `ReadValue -> [Gather] -> Concat -> Assign`
- a `PagedAttentionExtension` op
- a GPU `KVCache` op

This check is about the graph, not the use case. A speech model with a stateful text decoder is also an "LLM" here.

---

## Behavior Before 2026.5

| Model | IR's ASF is used? |
|---|---|
| Not an LLM | Yes |
| LLM | **No** |
| LLM with LoRA adapters (`lora_state_*` variables) on a GPU without XMX | Yes ([#34809](https://github.com/openvinotoolkit/openvino/pull/34809)) |

### The Problem

A model that really needs scaling cannot ask for it, because the plugin drops the value. One example is the decoder of **Fun-ASR-Nano**. It is a Qwen-style stateful decoder, so the plugin sees it as an LLM. Without scaling, its FP16 logits become NaN/Inf. The decoder then emits token 0 (`!`) at every step and never reaches EOS. Generation runs until `max_new_tokens`, which shows up as a very long inference or a timeout.

The decoder IR already has `ACTIVATIONS_SCALE_FACTOR=8.0`, and with this value the output is correct. But the plugin ignores it.

We could not simply apply the IR's ASF to all LLMs. All existing LLM IRs carry 8.0, so this would turn on scaling for every LLM. For some models, activations scaling changed the output or made inference slower.

And because every LLM IR carries 8.0, the plugin cannot tell from the value alone which model needs scaling.

---

## Behavior in 2026.5

The plugin now applies the IR's ASF to an LLM **when the value is not 8.0** ([#39233](https://github.com/openvinotoolkit/openvino/pull/39233)).

| Model | IR's ASF = 8.0 | IR's ASF = other value (e.g. 4.0) |
|---|---|---|
| Not an LLM | Used | Used |
| LLM | Ignored (same as before) | **Used (new)** |
| LLM with LoRA on a GPU without XMX | Used | Used |

- **Existing LLM IRs do not change.** They all carry 8.0 from Optimum Intel, and the plugin still ignores it. Their performance and accuracy stay the same.
- **A model that needs scaling can ask for it.** Export the IR with an ASF other than 8.0. The value 8.0 is treated as "default from the exporter", and any other value as "chosen for this model".
- **User settings still win.** `OV_ACTIVATIONS_SCALE_FACTOR` and the `ACTIVATIONS_SCALE_FACTOR` property work as before.

Optimum Intel now exports the Fun-ASR-Nano decoder with `ACTIVATIONS_SCALE_FACTOR=4.0` ([huggingface/optimum-intel#2091](https://github.com/huggingface/optimum-intel/pull/2091)).

### Result: Fun-ASR-Nano on GPU (FP16)

Measured with the 2026.5 plugin on Intel Arc Pro B70 (dGPU) and Intel UHD Graphics 770 (iGPU), Linux, FP16. The audio is 20 LibriSpeech test-clean samples, and `max_new_tokens` is 200. No user ASF is set: the value comes only from the IR. The encoder has no ASF in its IR and runs without scaling in both cases.

| Decoder IR's ASF | Effective ASF (decoder) | Output | WER | Time, 20 samples (B70 / UHD 770) |
|---|---|---|---|---|
| 8.0 | -1.0 (off) | `!` only, no EOS | 100% | 26.3 s / 128.8 s |
| 4.0 | 4.0 | Correct transcript | 0.73% | 6.3 s / 24.4 s |

With 8.0 every sample runs until `max_new_tokens`, so it is also much slower. Before 2026.5 the result is the same as the 8.0 row for both IRs, because the plugin ignores any ASF in an LLM IR.

*DISCLAIMER: The impact may vary depending on the system, model and usage.*

### What You Should Do

| You are | Action |
|---|---|
| Running an existing LLM IR | Nothing. Behavior does not change. |
| Exporting an LLM that overflows FP16 on GPU | Put an ASF other than 8.0 in the IR (for example 4.0). Check `compiled.get_property("ACTIVATIONS_SCALE_FACTOR")`. |
| Testing whether a model needs scaling | Run with `OV_ACTIVATIONS_SCALE_FACTOR=<value>` and compare with `-1`. If FP32 (`INFERENCE_PRECISION_HINT=f32`) is correct and FP16 is not, FP16 overflow is a likely cause. |

How to check for FP16 overflow quickly:

```bash
# Scaling off
OV_ACTIVATIONS_SCALE_FACTOR=-1 python your_script.py
# Scaling on
OV_ACTIVATIONS_SCALE_FACTOR=4 python your_script.py
```

---

## Summary

| | Before 2026.5 | 2026.5 |
|---|---|---|
| Non-LLM, IR's ASF | Used | Used |
| LLM, IR's ASF = 8.0 | Ignored | Ignored |
| LLM, IR's ASF != 8.0 | Ignored | **Used** |
| LLM with LoRA on GPU without XMX | Used | Used |
| User property or `OV_ACTIVATIONS_SCALE_FACTOR` | Used (highest priority) | Used (highest priority) |

---

## References

- [#27291](https://github.com/openvinotoolkit/openvino/pull/27291): per-layer activations scaling for FC to fix FP16 overflow
- [#27900](https://github.com/openvinotoolkit/openvino/pull/27900): read `runtime_options` from model rt_info
- [#28449](https://github.com/openvinotoolkit/openvino/pull/28449): read ASF from rt_info for non-LLMs
- [#28632](https://github.com/openvinotoolkit/openvino/pull/28632): disable activations scaling for LLMs on all platforms
- [#34809](https://github.com/openvinotoolkit/openvino/pull/34809): apply ASF to LLMs with LoRA on GPUs without XMX
- [#39233](https://github.com/openvinotoolkit/openvino/pull/39233): apply non-default ASF from rt_info to LLMs
- [huggingface/optimum-intel#2091](https://github.com/huggingface/optimum-intel/pull/2091): export the Fun-ASR-Nano decoder with ASF 4.0
