---
sidebar_position: 3
sidebar_label: "Legacy Features and Components"
description: "A list of deprecated OpenVINO™ components."
---

# Legacy Features and Components

Since OpenVINO has grown very rapidly in recent years, a number of its features
and components have been replaced by other solutions. Some of them are still
supported to assure OpenVINO users are given enough time to adjust their projects,
before the features are fully discontinued.

This section will give you an overview of these major changes and tell you how
you can proceed to get the best experience and results with the current OpenVINO
offering.

## Discontinued:

<details>
<summary>OpenVINO Development Tools Package</summary>

*New solution:* OpenVINO Runtime includes all supported components  
*Old solution:* [See how to install Development Tools](https://docs.openvino.ai/2024/documentation/legacy-features/install-dev-tools.html)  
  
OpenVINO Development Tools used to be the OpenVINO package with tools for
advanced operations on models, such as Model conversion API, Benchmark Tool,
Accuracy Checker, Annotation Converter, Post-Training Optimization Tool,
and Open Model Zoo tools. Most of these tools have been either removed,
replaced by other solutions, or moved to the OpenVINO Runtime package.

</details>

<details>
<summary>Model Optimizer / Conversion API</summary>

*New solution:* [Direct model support and OpenVINO Converter (OVC)](../openvino-workflow/model-preparation/index.md)  
*Old solution:* [Legacy Conversion API](https://docs.openvino.ai/2024/documentation/legacy-features/transition-legacy-conversion-api.html)  
  
The role of Model Optimizer and later the Conversion API was largely reduced
when all major model frameworks became supported directly. For converting model
files explicitly, it has been replaced with a more light-weight and efficient
solution, the OpenVINO Converter (launched with OpenVINO 2023.1).

</details>

<details>
<summary>Open Model ZOO</summary>

*New solution:* users are encouraged to use public model repositories such as [Hugging Face](https://huggingface.co/OpenVINO)  
*Old solution:* [Open Model ZOO](https://docs.openvino.ai/2024/documentation/legacy-features/model-zoo.html)  
  
Open Model ZOO provided a collection of models prepared for use with OpenVINO,
and a small set of tools enabling a level of automation for the process.
Since the tools have been mostly replaced by other solutions and several
other model repositories have recently grown in size and popularity,
Open Model ZOO will no longer be maintained. You may still use its resources
until they are fully removed. [Check the OMZ GitHub project](https://github.com/openvinotoolkit/open_model_zoo)

</details>

<details>
<summary>Multi-Device Execution</summary>

*New solution:* [Automatic Device Selection](../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection/index.md)  
*Old solution:* [Check the legacy solution](https://docs.openvino.ai/2024/documentation/legacy-features/multi-device.html)  
  
The behavior and results of the Multi-Device Execution mode are covered by the `CUMULATIVE_THROUGHPUT`
option of the Automatic Device Selection. The only difference is that `CUMULATIVE_THROUGHPUT` uses
the devices specified by AUTO, which means that adding devices manually is not mandatory,
while with MULTI, the devices had to be specified before the inference.

</details>

<details>
<summary>Caffe, and Kaldi model formats</summary>

*New solution:* conversion to ONNX via external tools  
*Old solution:* model support discontinued with OpenVINO 2024.0  
[The last version supporting Apache MXNet, Caffe, and Kaldi model formats](https://docs.openvino.ai/2023.3/mxnet_caffe_kaldi.html)  
[See the currently supported frameworks](../openvino-workflow/model-preparation/index.md)

</details>

<details>
<summary>Post-training Optimization Tool (POT)</summary>

*New solution:* Neural Network Compression Framework (NNCF) now offers the same functionality  
*Old solution:* POT discontinued with OpenVINO 2024.0  
[See how to use NNCF for model optimization](../openvino-workflow/model-optimization/index.md)  
[Check the NNCF GitHub project, including documentation](https://github.com/openvinotoolkit/nncf)

</details>

<details>
<summary>Inference API 1.0</summary>

*New solution:* API 2.0 launched in OpenVINO 2022.1  
*Old solution:* discontinued with OpenVINO 2024.0  
[2023.2 is the last version supporting API 1.0](https://docs.openvino.ai/archives/index.html#:~:text=2023.2,Release%20Notes)

</details>

<details>
<summary>Compile tool</summary>

*New solution:* the tool is no longer needed  
*Old solution:* discontinued with OpenVINO 2023.0  
If you need to compile a model for inference on a specific device, use the following script:

<div className="m-4">

<Tabs>
  <TabItem value="py" label="Python">
  ```python

  import openvino as ov

  core = ov.Core()

  compiled_model = core.compile_model(model_path, device, properties)
  output_stream = compiled_model.export_model()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;
  ov::CompiledModel model = core.compile_model("modelPath", "deviceName");
  std::fstream stream("compiled_model.blob");
  model.export_model(stream);
  ```
  </TabItem>
</Tabs>

</div>

</details>

<details>
<summary>TensorFlow integration (OVTF)</summary>

*New solution:* Direct model support and OpenVINO Converter (OVC)  
*Old solution:* discontinued in OpenVINO 2023.0  
  
OpenVINO now features a native TensorFlow support, with no need for explicit model
conversion.

</details>
