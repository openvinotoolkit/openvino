# Sync Benchmark C++ Sample {#openvino_inference_engine_samples_sync_benchmark_README}

This sample demonstrates how to estimate performace of a model using Synchronous Inference Request API. It makes sence to use synchronous inference only in latency oriented scenarios. Models with static input shapes are supported. Unlike [demos](@ref omz_demos) this sample doesn't have other configurable command line arguments. Feel free to modify sample's source code to try out different options.

The following C++ API is used in the application:

| Feature | API | Description |
| :--- | :--- | :--- |
| OpenVINO Runtime Version | `ov::get_openvino_version` | Get Openvino API version |
| Basic Infer Flow | `ov::Core`, `ov::Core::compile_model`, `ov::CompiledModel::create_infer_request`, `ov::InferRequest::get_tensor` | Common API to do inference: compile a model, create an infer request, configure input tensors |
| Synchronous Infer | `ov::InferRequest::infer` | Do synchronous inference |
| Model Operations | `ov::CompiledModel::inputs` | Get inputs of a model |
| Tensor Operations | `ov::Tensor::get_shape` | Get a tensor shape |
| Tensor Operations | `ov::Tensor::get_shape`, `ov::Tensor::data` | Get a tensor shape and its data. |

| Options | Values |
| :--- | :--- |
| Validated Models | [alexnet](@ref omz_models_model_alexnet), [googlenet-v1](@ref omz_models_model_googlenet_v1) [yolo-v3-tf](@ref omz_models_model_yolo_v3_tf), [face-detection-0200](@ref omz_models_model_face_detection_0200) |
| Model Format | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx) |
| Supported devices | [All](../../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md) |
| Other language realization | [Python](../../../python/benchmark/sync_benchmark/README.md) |

## How It Works

The sample compiles a model for a given device, randomly generates input data, performs synchronous inference multiple times for a given number of seconds. Then processes and reports performance results.

You can see the explicit description of
each sample step at [Integration Steps](../../../../docs/OV_Runtime_UG/integrate_with_your_application.md) section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../../docs/OV_Runtime_UG/Samples_Overview.md) section in OpenVINO™ Toolkit Samples guide.

## Running

```
sync_benchmark <path_to_model>
```

To run the sample, you need to specify a model:
- You can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).

> **NOTES**:
>
> - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

### Example

1. Install the `openvino-dev` Python package to use Open Model Zoo Tools:

```
python -m pip install openvino-dev[caffe]
```

2. Download a pre-trained model using:

```
omz_downloader --name googlenet-v1
```

3. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:

```
omz_converter --name googlenet-v1
```

4. Perform benchmarking using the `googlenet-v1` model on a `CPU`:

```
sync_benchmark googlenet-v1.xml
```

## Sample Output

The application outputs performance results.

```
[ INFO ] OpenVINO:
[ INFO ] Build ................................. <version>
[ INFO ] Count:      992 iterations
[ INFO ] Duration:   15009.8 ms
[ INFO ] Latency:
[ INFO ]        Median:     14.00 ms
[ INFO ]        Average:    15.13 ms
[ INFO ]        Min:        9.33 ms
[ INFO ]        Max:        53.60 ms
[ INFO ] Throughput: 66.09 FPS
```

## See Also

- [Integrate the OpenVINO™ Runtime with Your Application](../../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO™ Toolkit Samples](../../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
