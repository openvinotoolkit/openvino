# Benchmark Python Tool 

This page demonstrates how to use the Benchmark Python Tool to estimate deep learning inference performance on supported devices.

> **NOTE**: This page describes usage of the Python implementation of the Benchmark Tool. For the C++ implementation, refer to the [Benchmark C++ Tool](..\..\docs\articles_en\learn_openvino\openvino_samples\cpp_benchmark_tool.md) page. The Python version is recommended for benchmarking models that will be used in Python applications, and the C++ version is recommended for benchmarking models that will be used in C++ applications. Both tools have a similar command interface and backend.

For more detailed information on how this sample works, check the dedicated [article](..\..\docs\articles_en\learn_openvino\openvino_samples\python_benchmark_tool.md)

## Requriements

The Python benchmark_app is automatically installed when you install OpenVINO Developer Tools using [PyPI](..\..\docs\articles_en\get_started\installing-openvino-overview\installing-openvino-shared\installing-openvino-pip.md) Before running ``benchmark_app``, make sure the ``openvino_env`` virtual environment is activated, and navigate to the directory where your model is located.

The benchmarking application works with models in the OpenVINO IR (``model.xml`` and ``model.bin``) and ONNX (``model.onnx``) formats. 
Make sure to [convert your models](..\..\docs\articles_en\documentation\openvino_legacy_features\mo_ovc_transition\legacy_conversion_api.md) if necessary.
