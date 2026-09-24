# Throughput Benchmark Python Sample

This sample demonstrates how to estimate performance of a model using Asynchronous Inference Request API in throughput mode. You can configure the benchmark duration and minimum iteration count, or run continuously until a stop signal is received.

The reported results may deviate from what [benchmark_app](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/benchmark-tool.html) reports. One example is model input precision for computer vision tasks. benchmark_app sets uint8, while the sample uses default model precision which is usually float32.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/sync-benchmark.html)

## Usage

```sh
python throughput_benchmark.py <path_to_model> [device_name] [--seconds-to-run SECONDS] [--niter NITER]
```

The model path is required. The device defaults to `CPU`.

| Option | Description |
| --- | --- |
| `--seconds-to-run SECONDS` | Nonnegative integer duration in seconds. Defaults to `10`. Set to `0` to run until a stop signal. |
| `--niter NITER` | Positive integer minimum number of iterations. Defaults to `10`. |

For a finite duration, the benchmark runs until both the duration and minimum iteration count have been reached. For example, run for at least 30 seconds and 100 iterations:

```sh
python throughput_benchmark.py model.xml CPU --seconds-to-run 30 --niter 100
```

To run continuously:

```sh
python throughput_benchmark.py model.xml CPU --seconds-to-run 0
```

Press Ctrl+C (`SIGINT`) to stop, send `SIGTERM` on Linux or macOS, or press Ctrl+Break (`SIGBREAK`) on Windows. A stop signal ends the run even if `--niter` has not been reached.

## Requirements

| Options                        | Values                                                                                                                 |
| -------------------------------| -----------------------------------------------------------------------------------------------------------------------|
| Validated Models               | [yolo-v3-tf](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tf)                   |
|                                | [face-detection-0200](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-0200)  |
| Model Format                   | OpenVINO™ toolkit Intermediate Representation                                                                          |
|                                | (\*.xml + \*.bin), ONNX (\*.onnx)                                                                                      |
| Supported devices              | [All](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-devices.html)                    |
| Other language realization     | [C++](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/sync-benchmark.html)                               |

The following Python API is used in the application:

| Feature                   | API                                             | Description                                  |
| --------------------------| ------------------------------------------------|----------------------------------------------|
| OpenVINO API Version      | [openvino.\_\_version__]                          | Get Openvino API version.                    |
| Basic Infer Flow          | [openvino.Core],                        | Common API to do inference: compile a model, |
|                           | [openvino.Core.compile_model]           | configure input tensors.                     |
|                           | [openvino.InferRequest.get_tensor]      |                                              |
| Asynchronous Infer        | [openvino.AsyncInferQueue],             | Do asynchronous inference.                   |
|                           | [openvino.AsyncInferQueue.start_async], |                                              |
|                           | [openvino.AsyncInferQueue.wait_all],    |                                              |
|                           | [openvino.InferRequest.results]         |                                              |
| Model Operations          | [openvino.CompiledModel.inputs]         | Get inputs of a model.                       |
| Tensor Operations         | [openvino.Tensor.get_shape],            | Get a tensor shape and its data.             |
|                           | [openvino.Tensor.data]                  |                                              |

