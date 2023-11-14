# Automatic Speech Recognition C++ Sample

> **NOTE**: This sample is being deprecated and will no longer be maintained after OpenVINO 2023.2 (LTS). The main reason for it is the outdated state of the sample and its extensive usage of GNA, which is not going to be supported by OpenVINO beyond 2023.2. 

This sample demonstrates how to execute an Asynchronous Inference of acoustic model based on Kaldi\* neural networks and speech feature vectors.  

The sample works with Kaldi ARK or Numpy* uncompressed NPZ files, so it does not cover an end-to-end speech recognition scenario (speech to text), requiring additional preprocessing (feature extraction) to get a feature vector from a speech signal, as well as postprocessing (decoding) to produce text from scores.

For more detailed information on how this sample works, check the dedicated [article](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_automatic_speech_recognition.md)

## Requirements 

| Options                    | Values                                                                                                                                              |
| ---------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------------|
| Validated Models           | Acoustic model based on Kaldi\* neural networks (see                                                                                                |
|                            | [Model Preparation](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_automatic_speech_recognition.md#model-preparation-speech)  |
|                            | section)                                                                                                                                            |
| Model Format               | OpenVINO™ toolkit Intermediate Representation (*.xml + *.bin)                                                                                       |
| Supported devices          | See [Execution Modes](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_automatic_speech_recognition.md#execution-modes-speech)  |
|                            | section below and [List Supported Devices](..\..\..\docs\articles_en\about_openvino\compatibility_and_support\Supported_Devices.md)                 |

The following C++ API is used in the application:

| Feature                  | API                                                                           | Description                                                                  |
| -------------------------| ------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| Available Devices        | ``ov::Core::get_available_devices``, ``ov::Core::get_property``               | Get information of the devices for inference                                 |
| Import/Export Model      | ``ov::Core::import_model``, ``ov::CompiledModel::export_model``               | The GNA plugin supports loading and saving of the GNA-optimized model        |
| Model Operations         | ``ov::set_batch``, ``ov::Model::add_output``, ``ov::CompiledModel::inputs``,  |                                                                              |
|                          | ``ov::CompiledModel::outputs``                                                | Managing of model: configure batch_size, input and output tensors            |
| Node Operations          | ``ov::OutputVector::size``, ``ov::Output::get_shape``                         | Get node shape                                                               |
| Asynchronous Infer       | ``ov::InferRequest::start_async``, ``ov::InferRequest::wait``                 | Do asynchronous inference and waits until inference result becomes available |
| InferRequest Operations  | ``ov::InferRequest::query_state``, ``ov::VariableState::reset``               | Gets and resets CompiledModel state control                                  |
| Tensor Operations        | ``ov::Tensor::get_size``, ``ov::Tensor::data``,                               |                                                                              |
|                          | ``ov::InferRequest::get_tensor``                                              | Get a tensor, its size and data                                              |
| Profiling                | ``ov::InferRequest::get_profiling_info``                                      | Get infer request profiling info                                             |


Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_sample_hello_classification.md).
