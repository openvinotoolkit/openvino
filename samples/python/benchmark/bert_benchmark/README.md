# Bert Benchmark Python Sample

This sample demonstrates how to estimate performance of a Bert model using Asynchronous Inference Request API. Unlike [demos](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos) this sample doesn't have configurable command line arguments. Feel free to modify sample's source code to try out different options.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/bert-benchmark.html)

The sample downloads a model and a tokenizer, export the model to onnx, reads the exported model and reshapes it to enforce dynamic input shapes, compiles the resulting model, downloads a dataset and runs benchmarking on the dataset.

## Requirements

The following Python API is used in the application:

| Feature                  | API                                             | Description                                  |
| -------------------------| ------------------------------------------------|----------------------------------------------|
| OpenVINO API Version     | [openvino.\_\_version__]                          | Get Openvino API version.                    |
| Basic Infer Flow         | [openvino.runtime.Core],                        | Common API to do inference: compile a model. |
|                          | [openvino.runtime.Core.compile_model]           |                                              |
| Asynchronous Infer       | [openvino.runtime.AsyncInferQueue],             | Do asynchronous inference.                   |
|                          | [openvino.runtime.AsyncInferQueue.start_async], |                                              |
|                          | [openvino.runtime.AsyncInferQueue.wait_all]     |                                              |
| Model Operations         | [openvino.runtime.CompiledModel.inputs]         | Get inputs of a model.                       |
