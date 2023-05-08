# Model Preparation {#openvino_docs_model_processing_introduction}

@sphinxdirective
.. toctree::
   :maxdepth: 1
   :hidden:

   Supported_Model_Formats
   openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide
   omz_tools_downloader


Every deep learning workflow begins with obtaining a model. You can choose to prepare a custom one, use a ready-made solution and adjust it to your needs, or even download and run a pre-trained network from an online database, such as `TensorFlow Hub <https://tfhub.dev/>`__, `Hugging Face <https://huggingface.co/>`__, `Torchvision models <https://pytorch.org/hub/>`__.

:doc:`OpenVINO™ supports several model formats <Supported_Model_Formats>` and allows converting them to it's own, `openvino.runtime.Model <api/ie_python_api/_autosummary/openvino.runtime.Model.html>`__ or `ov.Model <api/ie_python_api/_autosummary/openvino.runtime.Model.html>`__, providing a tool dedicated to this task.

There are several options to convert a model from original framework to OpenVINO model format (``ov.Model``).

The ``read_model()`` method converts a model from original framework to ``ov.Model``. The resulting ``ov.Model`` can be inferred in the same training environment (python script or Jupiter Notebook).

Model Conversion API (the ``mo.convert_model()`` method) converts a model from original framework to ``ov.Model``. The resulting ``ov.Model`` can be inferred in the same training environment (python script or Jupiter Notebook). ``convert_model()`` also has a set of parameters to :doc:`cut the model <openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model>`, :doc:`set input shapes or layout <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`, :doc:`add preprocessing <openvino_docs_MO_DG_Additional_Optimization_Use_Cases>`, etc.

.. image:: _static/images/model_conversion_diagram.svg
   :alt: model conversion diagram

Another option to convert a model is to use MO command line tool. The resulting model is saved to IR (Intermediate Representation) which can be read with the ``ov.read_model()`` method, then be compiled and inferred.

``ov.Model`` can be serialized to IR using the ``ov.serialize()`` method. The serialized IR can be further optimized using :doc:`Post-Training Optimization tool <pot_introduction>`.

Conversion is not required for ONNX, PaddlePaddle, TensorFlow Lite and TensorFlow models (check :doc:`TensorFlow Frontend Capabilities and Limitations <openvino_docs_MO_DG_TensorFlow_Frontend>`), as OpenVINO provides C++ and Python APIs for importing them to OpenVINO Runtime directly. It provides a convenient way to quickly switch from framework-based code to OpenVINO-based code in your inference application.

This section describes how to obtain and prepare your model for work with OpenVINO to get the best inference results:

* :doc:`See the supported formats and how to use them in your project <Supported_Model_Formats>`.
* :doc:`Convert different model formats to the ov.Model format <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.


@endsphinxdirective
