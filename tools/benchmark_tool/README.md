# Benchmark Python Tool

This page demonstrates how to use the Benchmark Python Tool to estimate deep learning inference performance on supported devices.

> **NOTE**: This page describes usage of the Python implementation of the Benchmark Tool. For the C++ implementation, refer to the [Benchmark C++ Tool](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/benchmark-tool.html) page. The Python version is recommended for benchmarking models that will be used in Python applications, and the C++ version is recommended for benchmarking models that will be used in C++ applications. Both tools have a similar command interface and backend.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/benchmark-tool.html)

## Requirements

The Python benchmark_app is automatically installed when you install OpenVINO Developer Tools using [PyPI](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-pip.html) Before running ``benchmark_app``, make sure the ``openvino_env`` virtual environment is activated, and navigate to the directory where your model is located.

The benchmarking application works with models in the OpenVINO IR (``model.xml`` and ``model.bin``) and ONNX (``model.onnx``) formats.
Make sure to [convert your models](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-to-ir.html) if necessary.
