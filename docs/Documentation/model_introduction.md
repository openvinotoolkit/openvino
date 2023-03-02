# Model Preparation {#openvino_docs_model_processing_introduction}

@sphinxdirective
.. toctree::
   :maxdepth: 1
   :hidden:

   Supported_Model_Formats
   openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide
   omz_tools_downloader

@endsphinxdirective


Every deep learning workflow begins with obtaining a model. You can choose to prepare a custom one, use a ready-made solution and adjust it to your needs, or even download and run a pre-trained network from an online database, such as OpenVINO's [Open Model Zoo](../model_zoo.md).

[OpenVINO™ supports several model formats](../MO_DG/prepare_model/convert_model/supported_model_formats.md) and allows to convert them to it's own, OpenVINO IR, providing a tool dedicated to this task.

[Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) reads the original model and creates the OpenVINO IR model (.xml and .bin files) so that inference can ultimately be performed without delays due to format conversion. Optionally, Model Optimizer can adjust the model to be more suitable for inference, for example, by [alternating input shapes](../MO_DG/prepare_model/convert_model/Converting_Model.md), [embedding preprocessing](../MO_DG/prepare_model/Additional_Optimizations.md) and [cutting training parts off](../MO_DG/prepare_model/convert_model/Cutting_Model.md).

The approach to fully convert a model is considered the default choice, as it allows the full extent of OpenVINO features. The OpenVINO IR model format is used by other conversion and preparation tools, such as the Post-Training Optimization Tool, for further optimization of the converted model.

Conversion is not required for ONNX, PaddlePaddle, and TensorFlow models (check [TensorFlow Frontend Capabilities and Limitations](../resources/tensorflow_frontend.md)), as OpenVINO provides C++ and Python APIs for importing them to OpenVINO Runtime directly. It provides a convenient way to quickly switch from framework-based code to OpenVINO-based code in your inference application.

This section describes how to obtain and prepare your model for work with OpenVINO to get the best inference results:
* [See the supported formats and how to use them in your project](../MO_DG/prepare_model/convert_model/supported_model_formats.md)
* [Convert different model formats to the OpenVINO IR format](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
* [Automate model-related tasks with Model Downloader and additional OMZ Tools](https://docs.openvino.ai/latest/omz_tools_downloader.html).

To begin with, you may want to [browse a database of models for use in your projects](../model_zoo.md).
