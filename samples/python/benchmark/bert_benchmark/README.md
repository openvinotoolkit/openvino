# Bert Benchmark Python Sample

This sample demonstrates how to estimate performance of a Bert model using Asynchronous Inference Request API. Unlike [demos](https://docs.openvino.ai/2023.2/omz_demos.html) this sample doesn't have configurable command line arguments. Feel free to modify sample's source code to try out different options.

The following Python API is used in the application:

| Feature                  | API                                             | Description                                  |
| -------------------------| ------------------------------------------------|----------------------------------------------|
| OpenVINO Runtime Version | [openvino.runtime.get_version]                  | Get Openvino API version.                    |
| Basic Infer Flow         | [openvino.runtime.Core],                        | Common API to do inference: compile a model. |
|                          | [openvino.runtime.Core.compile_model]           |                                              |
| Asynchronous Infer       | [openvino.runtime.AsyncInferQueue],             | Do asynchronous inference.                   |
|                          | [openvino.runtime.AsyncInferQueue.start_async], |                                              |
|                          | [openvino.runtime.AsyncInferQueue.wait_all]     |                                              |
| Model Operations         | [openvino.runtime.CompiledModel.inputs]         | Get inputs of a model.                       |
