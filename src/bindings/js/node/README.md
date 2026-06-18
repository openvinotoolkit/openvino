# OpenVINO™ for Node.js

<div align="center">

![OpenVINO logo](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/assets/openvino-logo-purple-black.svg?raw=1)

[<b>Quick install</b>](#quick-install) •
[<b>Getting Started</b>](#getting-started) •
[<b>Use Cases</b>](#example-use-cases) •
[<b>Documentation</b>](https://docs.openvino.ai/2026/api/nodejs_api/nodejs_api.html) •
[<b>Samples</b>](https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/README.md)

[![NPM Version](https://img.shields.io/npm/v/openvino-node)](https://www.npmjs.com/package/openvino-node?activeTab=versions)
[![OS](https://img.shields.io/badge/OS-Linux_|_Windows_|_macOS-blue)](#supported-platforms)
[![NPM License](https://img.shields.io/npm/l/openvino-node)](#license)

</div>

**openvino-node** brings the [**OpenVINO™ Runtime**](https://docs.openvino.ai/) to **Node.js**. Deploy deep learning models with high-performance, hardware-accelerated inference directly from JavaScript and TypeScript applications. The package provides bindings to a subset of the OpenVINO Runtime API.

Prebuilt native addons and the OpenVINO runtime are fetched during `npm install`. You do not need a separate OpenVINO SDK installation for typical use.

## Key Features and Benefits:
 - 📦 Ready-to-use Runtime: Read, compile, and run OpenVINO IR, ONNX, TensorFlow, TFLite, and PaddlePaddle models directly from Node.js.
 - 📥 Plug-and-play install: Prebuilt native addons and the OpenVINO runtime are downloaded on `npm install` — no manual OpenVINO installation for typical use.
 - 🖥️ In-process inference: Run models directly inside your Node.js application process — no separate inference service or access tokens.
 - 🚀 Performance Optimization: Hardware-specific optimizations for CPU, GPU, and NPU devices, with synchronous and asynchronous inference.
 - 👨‍💻 Programming Language Support: API aligned with the OpenVINO Python/C++ APIs where possible, with TypeScript type definitions included.
 - 🧰 Preprocessing Support: Built-in pre/post-processing (layout, resize, element type conversion) via `PrePostProcessor`.

## Example use cases

**openvino-node** can run any model supported by OpenVINO Runtime. The following tasks are demonstrated by the bundled samples and notebooks:

- [Image classification](https://github.com/openvinotoolkit/openvino/tree/master/samples/js/node/hello_classification) - Recognize the main object in an image.
- [Asynchronous inference](https://github.com/openvinotoolkit/openvino/tree/master/samples/js/node/classification_sample_async) - Run multiple inference requests in parallel for higher throughput.
- [Object detection](https://github.com/openvinotoolkit/openvino/tree/master/samples/js/node/hello_reshape_ssd) - Detect and localize objects with bounding boxes (SSD), including model reshaping.
- [Optical character recognition (OCR)](https://github.com/openvinotoolkit/openvino/tree/master/samples/js/node/optical_character_recognition) - Detect and recognize text in images.
- [Image segmentation & background removal](https://github.com/openvinotoolkit/openvino/tree/master/samples/js/node/vision_background_removal) - Produce per-pixel masks to segment or remove the background.
- [Semantic segmentation](https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/notebooks/hello-segmentation.nnb) - Classify each pixel of an image into a category.
- [Pose estimation](https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/notebooks/pose-estimation.nnb) - Estimate human body keypoints.
- [Question answering (NLP)](https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/notebooks/question-answering.nnb) - Answer questions about a given text context.

## Quick install

```bash
npm install openvino-node
```

```js
const { addon: ov } = require("openvino-node");
```

## Requirements

Node.js **≥ 21**. Refer to the [supported platforms](#supported-platforms) for more details.

## Supported platforms

| **OS** | **x86** | **ARM** |
| ------ | ------- | ------- |
| **Windows** | ✅ | ❌ |
| **Linux** | ✅ | ✅ |
| **macOS** | ❌ | ✅ |

Prebuilt binaries are downloaded for your OS/arch during `npm install`. If a platform is unsupported, you can build from source per the [JavaScript API Developer Documentation](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/docs/README.md#openvino-node-package-developer-documentation).

## Getting started

### Model preparation

OpenVINO works best with models converted to **OpenVINO IR** (for example with [Optimum Intel](https://github.com/huggingface/optimum-intel) or `ovc`). ONNX, TensorFlow, TFLite, and PaddlePaddle models can also be read directly. See the [model preparation documentation](https://docs.openvino.ai/2026/openvino-workflow/model-preparation.html) for more details.

### Minimal example

Initialize the runtime `Core`, read a model, compile it for a device, and run inference:

```js
const { addon: ov } = require("openvino-node");

async function main() {
  const core = new ov.Core();

  // Read a model (OpenVINO IR, ONNX, TF, TFLite, or Paddle)
  const model = await core.readModel("/path/to/model.xml");

  // Compile the model for a device: "CPU", "GPU", or "NPU"
  const compiledModel = await core.compileModel(model, "CPU");

  // Create an infer request, set input, and run inference
  const inferRequest = compiledModel.createInferRequest();
  inferRequest.setInputTensor(inputTensor);
  inferRequest.infer();

  // Read the output
  const outputTensor = inferRequest.getTensor(compiledModel.outputs[0]);
  console.log(outputTensor.data);
}

main();
```

Refer to the complete description of the `addon` API in the
[documentation](https://docs.openvino.ai/2026/api/nodejs_api/addon.html).

More runnable examples are available in the [**OpenVINO Node.js samples**](https://github.com/openvinotoolkit/openvino/blob/master/samples/js/node/README.md).

## Usage in Electron applications

To use the package in development of Electron applications on Windows, make sure that the **Desktop
development with C++** component from
[Build Tools for Visual Studio](https://aka.ms/vs/17/release/vs_BuildTools.exe) is installed.

## Build From Sources

For more details, refer to the
[OpenVINO™ JavaScript API Developer Documentation](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/docs/README.md#openvino-node-package-developer-documentation)

## Contributing

Contributions are always welcome! Read the
[Contribution Guide](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/CONTRIBUTING.md)
to learn how you can get involved.

## License

The OpenVINO™ repository is licensed under [Apache License Version 2.0](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.
