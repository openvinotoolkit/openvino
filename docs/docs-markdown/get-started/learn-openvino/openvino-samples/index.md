---
sidebar_position: 2
sidebar_label: "OpenVINO™ Samples"
description: "OpenVINO™ samples include a collection of simple console applications that explain how to implement the capabilities and features of OpenVINO API into an application."
---

# OpenVINO™ Samples

(code samples)=


The OpenVINO™ samples are simple console applications that show how to utilize
specific OpenVINO API capabilities within an application. They can assist you in
executing specific tasks such as loading a model, running inference, querying
specific device capabilities, etc.

The applications include:

:::info[Important]
All C++ samples support input paths containing only ASCII characters, except
for the Hello Classification Sample, which supports Unicode.
:::

- [Hello Classification Sample](hello-classification.md) -
  Inference of image classification networks like AlexNet and GoogLeNet using
  Synchronous Inference Request API. Input of any size and layout can be set to
  an infer request which will be pre-processed automatically during inference.
  The sample supports only images as input and supports input paths containing
  only Unicode characters.
- [Hello NV12 Input Classification Sample](hello-nv12-input-classification.md) -
  Input of any size and layout can be provided to an infer request. The sample
  transforms the input to the NV12 color format and pre-process it automatically
  during inference. The sample supports only images as input.
- [Hello Query Device Sample](hello-query-device.md) -
  Query of available OpenVINO devices and their metrics, configuration values.
- [Hello Reshape SSD Sample](hello-reshape-ssd.md) -
  Inference of SSD networks resized by ShapeInfer API according to an input size.
- [Image Classification Async Sample](image-classification-async.md) -
  Inference of image classification networks like AlexNet and GoogLeNet using
  Asynchronous Inference Request API. The sample supports only images as inputs.
- [OpenVINO Model Creation Sample](model-creation.md) -
  Construction of the LeNet model using the OpenVINO model creation sample.
- **Benchmark Samples** - Simple estimation of a model inference performance
  
  - [Sync Samples](sync-benchmark.md)
  - [Throughput Samples](throughput-benchmark.md)
  - [Bert Python Sample](bert-benchmark.md)
- [Benchmark Application](benchmark-tool.md) - Estimates deep
  learning inference performance on supported devices for synchronous and
  asynchronous modes.
  
  Python version of the benchmark tool is a core component of the OpenVINO
  installation package and may be executed with the following command:
  
  ```console
  benchmark_app -m <model> -i <input> -d <device>
  ```

## Additional Resources

- [Get Started with Samples](get-started-demos.md)
- [OpenVINO Runtime User Guide](../../../openvino-workflow/running-inference/index.md)
