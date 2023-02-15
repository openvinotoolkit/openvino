# Throughput Benchmark Python* Sample {#openvino_inference_engine_ie_bridges_python_sample_throughput_benchmark_README}

This sample demonstrates how to estimate performace of a model using Asynchronous Inference Request API in throughput mode. Unlike [demos](@ref omz_demos) this sample doesn't have other configurable command line arguments. Feel free to modify sample's source code to try out different options.

The reported results may deviate from what [benchmark_app](../../../../tools/benchmark_tool/README.md) reports. One example is model input precision for computer vision tasks. benchmark_app sets uint8, while the sample uses default model precision which is usually float32.

The following Python\* API is used in the application:

| Feature | API | Description |
| :--- | :--- | :--- |
| OpenVINO Runtime Version | [openvino.runtime.get_version] | Get Openvino API version |
| Basic Infer Flow | [openvino.runtime.Core], [openvino.runtime.Core.compile_model], [openvino.runtime.InferRequest.get_tensor] | Common API to do inference: compile a model, configure input tensors |
| Asynchronous Infer | [openvino.runtime.AsyncInferQueue], [openvino.runtime.AsyncInferQueue.start_async], [openvino.runtime.AsyncInferQueue.wait_all], [openvino.runtime.InferRequest.results] | Do asynchronous inference |
| Model Operations | [openvino.runtime.CompiledModel.inputs] | Get inputs of a model |
| Tensor Operations | [openvino.runtime.Tensor.get_shape], [openvino.runtime.Tensor.data] | Get a tensor shape and its data. |

| Options | Values |
| :--- | :--- |
| Validated Models | [alexnet](@ref omz_models_model_alexnet), [googlenet-v1](@ref omz_models_model_googlenet_v1) [yolo-v3-tf](@ref omz_models_model_yolo_v3_tf), [face-detection-0200](@ref omz_models_model_face_detection_0200) |
| Model Format | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx) |
| Supported devices | [All](../../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md) |
| Other language realization | [C++](../../../cpp/benchmark/throughput_benchmark/README.md) |

## How It Works

The sample compiles a model for a given device, randomly generates input data, performs asynchronous inference multiple times for a given number of seconds. Then processes and reports performance results.

You can see the explicit description of
each sample step at [Integration Steps](../../../../docs/OV_Runtime_UG/integrate_with_your_application.md) section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Running

```
python throughput_benchmark.py <path_to_model>
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
python throughput_benchmark.py googlenet-v1.xml
```

## Sample Output

The application outputs performance results.

```
[ INFO ] OpenVINO:
[ INFO ] Build ................................. <version>
[ INFO ] Count:          2817 iterations
[ INFO ] Duration:       10012.65 ms
[ INFO ] Latency:
[ INFO ]     Median:     13.80 ms
[ INFO ]     Average:    14.10 ms
[ INFO ]     Min:        8.35 ms
[ INFO ]     Max:        28.38 ms
[ INFO ] Throughput: 281.34 FPS
```

## See Also

- [Integrate the OpenVINO™ Runtime with Your Application](../../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO™ Toolkit Samples](../../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
