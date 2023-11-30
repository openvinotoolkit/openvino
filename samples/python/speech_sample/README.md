# Automatic Speech Recognition Python Sample

> **NOTE**: This sample is being deprecated and will no longer be maintained after OpenVINO 2023.2 (LTS). The main reason for it is the outdated state of the sample and its extensive usage of GNA, which is not going to be supported by OpenVINO beyond 2023.2. 

This sample demonstrates how to do a Synchronous Inference of acoustic model based on Kaldi\* neural models and speech feature vectors.

The sample works with Kaldi ARK or Numpy* uncompressed NPZ files, so it does not cover an end-to-end speech recognition scenario (speech to text), requiring additional preprocessing (feature extraction) to get a feature vector from a speech signal, as well as postprocessing (decoding) to produce text from scores.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2023.2/openvino_inference_engine_ie_bridges_python_sample_speech_sample_README.html)

## Requirements

| Options                     | Values                                                                                                                                                        |
| ----------------------------| --------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Validated Models            | Acoustic model based on Kaldi* neural models (see                                                                                                             |
|                             | [Model Preparation](https://docs.openvino.ai/2023.2/openvino_inference_engine_ie_bridges_python_sample_speech_sample_README.html#model-preparation) section)  |
| Model Format                | OpenVINO™ toolkit Intermediate Representation (.xml + .bin)                                                                                                   |
| Supported devices           | See [Execution Modes](https://docs.openvino.ai/2023.2/openvino_inference_engine_ie_bridges_python_sample_speech_sample_README.html#execution-modes)           |
|                             | section below and [List Supported Devices](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)                      |
| Other language realization  | [C++](https://docs.openvino.ai/2023.2/openvino_inference_engine_samples_speech_sample_README.html)                                                            |

Automatic Speech Recognition Python sample application demonstrates how to use the following Python API in applications:

| Feature                  | API                                                                                                                                                                                                             | Description                                                           |
| -------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Import/Export Model      | [openvino.runtime.Core.import_model](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.import_model),                                             |                                                                       |
|                          | [openvino.runtime.CompiledModel.export_model](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html#openvino.runtime.CompiledModel.export_model)                   | The GNA plugin supports loading and saving of the GNA-optimized model |
| Model Operations         | [openvino.runtime.Model.add_outputs](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.add_outputs) ,                                           |                                                                       |
|                          | [openvino.runtime.set_batch](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.html#openvino.runtime.set_batch),                                                                  |                                                                       |
|                          | [openvino.runtime.CompiledModel.inputs](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html#openvino.runtime.CompiledModel.inputs),                              |                                                                       |
|                          | [openvino.runtime.CompiledModel.outputs](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html#openvino.runtime.CompiledModel.outputs),                            |                                                                       |
|                          | [openvino.runtime.ConstOutput.any_name](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.ConstOutput.html#openvino.runtime.ConstOutput.any_name)                                 | Managing of model: configure batch_size, input and output tensors     |
| Synchronous Infer        | [openvino.runtime.CompiledModel.create_infer_request](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.CompiledModel.html#openvino.runtime.CompiledModel.create_infer_request),  |                                                                       |
|                          | [openvino.runtime.InferRequest.infer](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.InferRequest.html#openvino.runtime.InferRequest.infer)                                    | Do synchronous inference                                              |
| InferRequest Operations  | [openvino.runtime.InferRequest.get_input_tensor](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.InferRequest.html#openvino.runtime.InferRequest.get_input_tensor),             |                                                                       |
|                          | [openvino.runtime.InferRequest.model_outputs](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.InferRequest.html#openvino.runtime.InferRequest.model_outputs),                   |                                                                       |
|                          | [openvino.runtime.InferRequest.model_inputs](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.InferRequest.html#openvino.runtime.InferRequest.model_inputs),                     | Get info about model using infer request API                          |
| InferRequest Operations  | [openvino.runtime.InferRequest.query_state](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.InferRequest.html#openvino.runtime.InferRequest.query_state),                       |                                                                       |
|                          | [openvino.runtime.VariableState.reset](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.inference_engine.VariableState.html#openvino.inference_engine.VariableState.reset)               | Gets and resets CompiledModel state control                           |
| Profiling                | [openvino.runtime.InferRequest.profiling_info](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.InferRequest.html#openvino.runtime.InferRequest.profiling_info),                 |                                                                       |
|                          | [openvino.runtime.ProfilingInfo.real_time](https://docs.openvino.ai/2023.2/api/ie_python_api/_autosummary/openvino.runtime.ProfilingInfo.html#openvino.runtime.ProfilingInfo.real_time)                         | Get infer request profiling info                                      |

Basic OpenVINO™ Runtime API is covered by [Hello Classification Python* Sample](https://docs.openvino.ai/2023.2/openvino_inference_engine_ie_bridges_python_sample_hello_classification_README.html).
