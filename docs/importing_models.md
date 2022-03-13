# Introduction {#openvino_docs_importing_models_introduction}

This chapter provides you with the information about available capabilities to get you model imported to OpenVINO™.

Before starting inference with the OpenVINO™ toolkit, a pre-trained model should be exported into one of the supported model formats. Many models represented in ONNX, TensorFlow, PaddlePaddle, MXnet, Caffe, and Kaldi formats are supported in OpenVINO. Dedicated APIs and command-line tools are used to import such models to OpenVINO.

The process of importing is also referred to as model conversion because the original model is being converted to the model format used in OpenVINO. Converted model can be either:

*	Stored in the OpenVINO intermediate representation (IR) format: a pair of .xml and .bin files used for topology description and trained parameters respectively.
*	Used for inference immediately in runtime after conversion.

![](MO_DG/img/BASIC_FLOW_MO_simplified.svg)

To facilitate conversion to the IR file format, the [Model Optimizer](MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) tool is used. It reads the original model represented in one of the supported formats and produces IR files (.xml and .bin files). The main purpose of Model Optimizer is to prepare the model in an appropriate format for faster inference offline without spending time on model file conversion in the final inference application. Model Optimizer also can optionally adjust the model to be more suitable for inference, for example, by alternating input shapes, embedding preprocessing and cutting training parts off.

Besides conversion to the IR format, OpenVINO provides C++ and Python APIs for importing [ONNX](OV_Runtime_UG/ONNX_Support.md) and [PaddlePaddle](OV_Runtime_UG/Paddle_Support.md) models directly to OpenVINO™ Runtime without storing them in the IR file format. It provides a convenient way to quickly switch from framework-based code to OpenVINO-based code in your inference application. This way is available only for the ONNX and PaddlePaddle model formats. Use the Model Optimizer command-line tool to import models from other frameworks.

OpenVINO focuses on deployment scenarios, so the offline model conversion using the Model Optimizer tool is considered as the main path. IR files are also used as an input for other conversion and preparation tools. For example, you can use the Post-Training Optimization Tool for further optimization of the converted model. For more info about POT, read the [Post-Training Optimization Tool documentation](../tools/pot/README.md).
