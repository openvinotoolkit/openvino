---
name: ov-gguf
description: >
  Develop and troubleshoot OpenVINO GGUF conversion: model and mmproj support,
  architecture/projector extensions, ggml operation translators, and numerical accuracy
  in the native frontend or llama.cpp OpenVINO backend. Excludes general llama.cpp usage
  and unrelated build or CI failures.
---

Select the task below and read only the matching resource or section. Load another route
only when the task crosses that boundary; do not read all references upfront.

| Task | Read |
|---|---|
| Check model support | [Supported models](../../../src/frontends/gguf/docs/supported_models.md), or [supported projectors](../../../src/frontends/gguf/docs/mmproj.md#supported-projectors) for mmproj; include the relevant validation limits |
| Load an existing model or adapt its inputs/state | [Frontend usage](../../../src/frontends/gguf/README.md#loading-a-model); follow its adaptation links as needed |
| Add a decoder architecture, recurrent family, or whole-model builder | [Architecture support](references/architectures.md) |
| Add or replace an mmproj vision/audio encoder or projector component | [Projector support](references/projectors.md) |
| Implement a missing ggml operation, converter, or semantic `op_case` | [Operation translation](references/operations.md) |
| Diagnose wrong logits/features, generation drift, or a decode-only shape failure | [Accuracy debugging](references/accuracy.md) |

Identify whether the model uses native `.gguf` conversion or a supplied `GgufDecoder`
(llama.cpp cgraph). They share translators; their builders and validation coverage differ.
An unsupported quantization format belongs to the weight loader under `src/frontends/gguf/src/quant/`.
For support-only questions, answer from the relevant support list and evidence without loading
implementation routes. Successful registration or conversion alone does not establish accuracy.
