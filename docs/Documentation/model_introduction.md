# Model Preparation {#openvino_docs_model_processing_introduction}

@sphinxdirective
.. toctree::
   :maxdepth: 1
   :hidden:

   Supported_Model_Formats
   openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide
   omz_tools_downloader


Every deep learning workflow begins with obtaining a model. You can choose to prepare a custom one, use a ready-made solution and adjust it to your needs, or even download and run a pre-trained network from an online database, such as OpenVINO's :doc:`Open Model Zoo <model_zoo>`.

:doc:`OpenVINO™ supports several model formats <Supported_Model_Formats>` and allows to convert them to it's own, OpenVINO IR, providing a tool dedicated to this task.

:doc:`Model Optimizer <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>` reads the original model and creates the OpenVINO IR model (.xml and .bin files) so that inference can ultimately be performed without delays due to format conversion. Optionally, Model Optimizer can adjust the model to be more suitable for inference, for example, by :doc:`alternating input shapes <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`, :doc:`embedding preprocessing <openvino_docs_MO_DG_Additional_Optimization_Use_Cases>` and :doc:`cutting training parts off <openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model>`.

The approach to fully convert a model is considered the default choice, as it allows the full extent of OpenVINO features. The OpenVINO IR model format is used by other conversion and preparation tools, such as the Post-Training Optimization Tool, for further optimization of the converted model.

Conversion is not required for ONNX, PaddlePaddle, TensorFlow Lite and TensorFlow models (check :doc:`TensorFlow Frontend Capabilities and Limitations <openvino_docs_MO_DG_TensorFlow_Frontend>`), as OpenVINO provides C++ and Python APIs for importing them to OpenVINO Runtime directly. It provides a convenient way to quickly switch from framework-based code to OpenVINO-based code in your inference application.

This section describes how to obtain and prepare your model for work with OpenVINO to get the best inference results:

* :doc:`See the supported formats and how to use them in your project <Supported_Model_Formats>`.
* :doc:`Convert different model formats to the OpenVINO IR format <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
* :doc:`Automate model-related tasks with Model Downloader and additional OMZ Tools <omz_tools_downloader>`.

To begin with, you may want to :doc:`browse a database of models for use in your projects <model_zoo>`.

@endsphinxdirective
