# Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository {#openvino_docs_install_guides_installing_openvino_pip}

You can install Intel® Distribution of OpenVINO™ toolkit through the PyPI repository, including both OpenVINO™ Runtime and OpenVINO™ Development Tools. Besides, from the 2022.1 release, OpenVINO Development Tools can only be installed via PyPI.


## Installing OpenVINO Runtime

The OpenVINO Runtime contains a set of libraries for an easy inference integration into your applications and supports heterogeneous execution across Intel® CPU and Intel® GPU hardware. To install OpenVINO Runtime, use the following command:
```
pip install openvino
```

For system requirements and more detailed steps, see <https://pypi.org/project/openvino>.


## Installing OpenVINO Development Tools

OpenVINO Development Tools include Model Optimizer, Benchmark Tool, Accuracy Checker, Post-Training Optimization Tool and Open Model Zoo tools including Model Downloader. While installing OpenVINO Development Tools, OpenVINO Runtime will also be installed as a dependency, so you don't need to install OpenVINO Runtime separately.

Use the following command to install OpenVINO Development Tools:
```
pip install openvino-dev[EXTRAS]
```
where the EXTRAS parameter specifies one or more deep learning frameworks via these values: `caffe`, `kaldi`, `mxnet`, `onnx`, `pytorch`, `tensorflow`, `tensorflow2`. Make sure that you install the corresponding frameworks for your models.

For example, to install and configure the components for working with TensorFlow 2.x, MXNet and Caffe, use the following command:
```
pip install openvino-dev[tensorflow2,mxnet,caffe]
```

> **NOTE**: For TensorFlow, use the `tensorflow2` value as much as possible. The `tensorflow` value is provided only for compatibility reasons.
   
For system requirements and more detailed steps, see <https://pypi.org/project/openvino-dev>.


## Additional Resources

- [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit)
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- [Inference Engine Developer Guide](../OV_Runtime_UG/Deep_Learning_Inference_Engine_DevGuide.md)
- [Inference Engine Samples Overview](../OV_Runtime_UG/Samples_Overview.md)
