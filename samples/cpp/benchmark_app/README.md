# Benchmark C++ Tool

This page demonstrates how to use the Benchmark C++ Tool to estimate deep learning inference performance on supported devices.

> **NOTE**: This page describes usage of the C++ implementation of the Benchmark Tool. For the Python implementation, refer to the [Benchmark Python Tool](..\..\..\docs\articles_en\learn_openvino\openvino_samples\python_benchmark_tool.md) page. The Python version is recommended for benchmarking models that will be used in Python applications, and the C++ version is recommended for benchmarking models that will be used in C++ applications. Both tools have a similar command interface and backend.

For more detailed information on how this sample works, check the dedicated [article](..\..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_benchmark_tool.md)

## Requriements

To use the C++ benchmark_app, you must first build it following the [Build the Sample Applications](..\..\..\docs\articles_en\learn_openvino\openvino_samples.md) instructions and then set up paths and environment variables by following the [Get Ready for Running the Sample Applications](..\..\..\docs\articles_en\learn_openvino\openvino_samples\get_started_demos.md) instructions. Navigate to the directory where the benchmark_app C++ sample binary was built.

> **NOTE**: If you installed OpenVINO Runtime using PyPI or Anaconda Cloud, only the [Benchmark Python Tool](..\..\..\docs\articles_en\learn_openvino\openvino_samples\python_benchmark_tool.md) is available, and you should follow the usage instructions on that page instead.

The benchmarking application works with models in the OpenVINO IR, TensorFlow, TensorFlow Lite, PaddlePaddle, PyTorch and ONNX formats. If you need it, OpenVINO also allows you to [convert your models](..\..\..\docs\articles_en\documentation\openvino_legacy_features\mo_ovc_transition\legacy_conversion_api.md).
