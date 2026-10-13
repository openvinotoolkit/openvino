---
sidebar_position: 2
sidebar_label: "Performance Information F.A.Q."
description: "Check the F.A.Q. for performance benchmarks in Intel® Distribution of OpenVINO™ toolkit."
---

# Performance Information F.A.Q.

<details>
<summary>How often do performance benchmarks get updated?</summary>

New performance benchmarks are typically published on every
`major.minor` release of the Intel® Distribution of OpenVINO™ toolkit.

</details>

<details>
<summary>Where can I find the models used in the performance benchmarks?</summary>

All models used are published on [Hugging Face](https://huggingface.co/OpenVINO).

</details>

<details>
<summary>Will there be any new models added to the list used for benchmarking?</summary>

The models used in the performance benchmarks were chosen based
on general adoption and usage in deployment scenarios. New models that
support a diverse set of workloads and usage are added periodically.

</details>

<details>
<summary>How can I run the benchmark results on my own?</summary>

All of the performance benchmarks on traditional network models are generated using the
open-source tool within the Intel® Distribution of OpenVINO™ toolkit
called [benchmark_app](../../get-started/learn-openvino/openvino-samples/benchmark-tool.md).

For diffusers (Stable-Diffusion) and foundational models (aka LLMs) please use the OpenVINO GenAI
opensource repo [OpenVINO GenAI tools/llm\_bench](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench)

For a simple instruction on testing performance, see the [Getting Performance Numbers Guide](getting-performance-numbers.md).

</details>

<details>
<summary>Where can I find a more detailed description of the workloads used for benchmarking?</summary>

The image size used in inference depends on the benchmarked
network. The table below presents the list of input sizes for each
network model and a link to more information on that model:

| Model | Public Network | Task | Input Size |
| --- | --- | --- | --- |
| [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | DeepSeek, HF | Auto regressive language | 128K |
| [Gemma-3n-e4b-it](https://huggingface.co/google/gemma-3n-E4B-it) | Hugginface | Multimodal, instruction tuned for low-resource use cases. | 32K |
| [Gemma-4-e2b-it](https://huggingface.co/google/gemma-4-E2B-it) | Hugginface | Multimodal, instruction tuned for low-resource use cases. | 128K |
| [Gemma-4-26b-a4b](https://huggingface.co/google/gemma-4-26B-A4B-it) | Hugginface | Multimodal MoE, instruction tuned with fast response time. | 256K |
| [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) | Hugginface | Reasoning, agentic and developer use cases. | 128K |
| [Kokoro-82m](https://huggingface.co/hexgrad/Kokoro-82M) | Hexgrad | Lightweight text-to-speech open-weight model. Multi-lingual, including accents. | 128K |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | Meta AI | Auto regressive language | 128K |
| [MiniCPM-v-2\_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) | OpenBMB | Multimodal, image-text to text | 32K |
| [Mistral-7B-Instruct-V0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Mistral AI | Auto regressive language | 32K |
| [Phi4-mini-Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | Hugginface | Auto regressive language | 4096 |
| [Qwen-3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Huggingface | Multimodal, Gated Delta Networks, Sparse MoE | 256K |
| [Qwen-3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Huggingface | Causal Language with Vision Encoder | 256K (up to 1M) |
| [Stable-Diffusion-V1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | Hugginface | Latent Diffusion Model | 77 |
| [LTX-VIDEO](https://huggingface.co/Lightricks/LTX-Video) | Hugginface | DiT based video generation. Diffusion Transformer. | up to 300 |
| [bert-base-cased](https://github.com/PaddlePaddle/PaddleNLP/tree/v2.1.1) | BERT | question / answer | 128 |
| [Detectron-V2](https://huggingface.co/kalinkov/Detectron2_FasterRCNN_R50_FPN_InsectDetection/tree/main) | Detectron-V2 | object instance segmentation | 800x800 |
| [mobilenet-v2](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v2-pytorch) | Mobilenet V2 PyTorch | classification | 224x224 |
| [resnet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) | ResNet-50\_v1\_ILSVRC-2012 | classification | 224x224 |
| [ssd-resnet34-1200-onnx](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd-resnet34-1200-onnx) | ssd-resnet34 onnx model | object detection | 1200x1200 |
| [yolov11](https://github.com/ultralytics/ultralytics) | Yolov11 | object detection | 640x640 |
| [yolov26n](https://huggingface.co/openvision/yolo26-n) | Yolov26 Nano | object detection | 640x640 |

</details>

<details>
<summary>Where can I purchase the specific hardware used in the benchmarking?</summary>

Intel partners with vendors all over the world. For a list of Hardware Manufacturers, see the
[Intel® AI: In Production Partners & Solutions Catalog](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/partners-solutions-catalog.html).
For more details, see the [Supported Devices](../../documentation/compatibility-and-support/supported-devices.md) article.

</details>

<details>
<summary>How can I optimize my models for better performance or accuracy?</summary>

Set of guidelines and recommendations to optimize models are available in the
[optimization guide](../../openvino-workflow/running-inference/optimize-inference/index.md).
Join the conversation in the [Community Forum](https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit) for further support.

</details>

<details>
<summary>Why are INT8 optimized models used for benchmarking on CPUs with no VNNI support?</summary>

The benefit of low-precision optimization extends beyond processors supporting VNNI
through Intel® DL Boost. The reduced bit width of INT8 compared to FP32
allows Intel® CPU to process the data faster. Therefore, it offers
better throughput on any converted model, regardless of the
intrinsically supported low-precision optimizations within Intel®
hardware. For comparison on boost factors for different network models
and a selection of Intel® CPU architectures, including AVX-2 with Intel®
Core™ i7-8700T, and AVX-512 (VNNI) with Intel® Xeon® 5218T and Intel®
Xeon® 8270, refer to the [Model Accuracy for INT8 and FP32 Precision](model-accuracy-int8-fp32.md)

</details>

<details>
<summary>How is Latency measured?</summary>

Latency is measured by running the OpenVINO™ Runtime in
synchronous mode. In this mode, each frame or image is processed through
the entire set of stages (pre-processing, inference, post-processing)
before the next frame or image is processed. This KPI is relevant for
applications where the inference on a single image is required. For
example, the analysis of an ultra sound image in a medical application
or the analysis of a seismic image in the oil & gas industry. Other use
cases include real or near real-time applications, e.g. the response of
industrial robot to changes in its environment and obstacle avoidance
for autonomous vehicles, where a quick response to the result of the
inference is required.

</details>

<link rel="stylesheet" type="text/css" href="../../_static/css/benchmark-banner.css"/>

Results may vary. For more information, see:
[Platforms, Configurations, Methodology](index.md),
[Legal Information](../additional-resources/terms-of-use.md).
