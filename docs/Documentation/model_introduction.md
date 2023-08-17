# Model Preparation {#openvino_docs_model_processing_introduction}

@sphinxdirective

.. meta::
   :description: Preparing models for OpenVINO Runtime. Learn how to convert and compile models from different frameworks or read them directly.


.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_convert_model_MO_OVC_transition
   Supported_Model_Formats
   openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide
   omz_tools_downloader


Every deep learning workflow begins with obtaining a model. You can choose to prepare a custom one, use a ready-made solution and adjust it to your needs, or even download and run a pre-trained network from an online database, such as `TensorFlow Hub <https://tfhub.dev/>`__, `Hugging Face <https://huggingface.co/>`__, `Torchvision models <https://pytorch.org/hub/>`__.

:doc:`OpenVINO™ supports several model formats <Supported_Model_Formats>` and allows converting them to it's own, `openvino.Model <api/ie_python_api/_autosummary/openvino.Model.html>`__ (`ov.Model <api/ie_python_api/_autosummary/openvino.runtime.Model.html>`__ ), providing a tool dedicated to this task.


.. note::

   In most of the Python tutorials and Python examples ``openvino`` namespace is replaced with short alias ``ov``.

   Example:

  .. code-block:: py

     import openvino as ov


There are several options to convert a model from original framework to OpenVINO model format (``ov.Model``).

The ``read_model()`` method reads a model from a file and produces ``ov.Model``. If the file is in one of the supported original framework file formats, it is converted automatically to OpenVINO Intermediate Representation. If the file is already in the OpenVINO IR format, it is read "as-is", without any conversion involved. ``ov.Model`` can be serialized to IR using the ``ov.serialize()`` method. The serialized IR can be further optimized using :doc:`Neural Network Compression Framework (NNCF) <ptq_introduction>` that applies post-training quantization methods.

Convert a model in Python
######################################

Model conversion API, specifically, the ``ov.convert_model()`` method converts a model from original framework to ``ov.Model``. ``ov.convert_model()`` returns ``ov.Model`` object in memory so the ``read_model()`` method is not required. The resulting ``ov.Model`` can be inferred in the same training environment (python script or Jupiter Notebook). ``ov.convert_model()`` provides a convenient way to quickly switch from framework-based code to OpenVINO-based code in your inference application. In addition to model files, ``ov.convert_model()`` can take OpenVINO extension objects constructed directly in Python for easier conversion of operations that are not supported in OpenVINO.

.. image:: _static/images/model_conversion_diagram.svg
   :alt: model conversion diagram

Convert a model with ``ovc`` (OpenVino Conversion) command-line tool
####################################################################

Another option to convert a model is to use ``ovc`` command-line tool. ``ovc`` is a cross-platform tool that facilitates the transition between training and deployment environments, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices in the same measure, as the ``ov.convert_model`` method.

``ovc`` requires the use of a pre-trained deep learning model in one of the supported formats: TensorFlow, TensorFlow Lite, PaddlePaddle, or ONNX. ``ovc`` converts the model to the OpenVINO Intermediate Representation format (IR), which needs to be read with the ``ov.read_model()`` method. Then, you can compile and infer the ``ov.Model`` later with :doc:`OpenVINO™ Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.


The figure below illustrates the typical workflow for deploying a trained deep learning model:

#TODO: Update BASIC_FLOW_MO_simplified.svg and replace 'mo' with 'ovc'
.. image:: _static/images/BASIC_FLOW_MO_simplified.svg

where IR is a pair of files describing the model:

* ``.xml`` - Describes the network topology.
* ``.bin`` - Contains the weights and biases binary data.


Model files (not Python objects) from ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite  (check :doc:`TensorFlow Frontend Capabilities and Limitations <openvino_docs_MO_DG_TensorFlow_Frontend>`) do not require a separate step for model conversion, that is ``ov.convert_model``. OpenVINO provides C++ and Python APIs for importing the models to OpenVINO Runtime directly by just calling the ``read_model`` method. 

The results of both ``ovc`` and ``ov.convert_model()`` conversion methods described above are the same. You can choose one of them, depending on what is most convenient for you. Keep in mind that there should not be any differences in the results of model conversion if the same set of parameters is used.

This section describes how to obtain and prepare your model for work with OpenVINO to get the best inference results:

* :doc:`See the supported formats and how to use them in your project <Supported_Model_Formats>`.
* :doc:`Convert different model formats to the ov.Model format <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
* :doc:`Transition guide from MO / ov.tools.mo.convert_model() to OVC / ov.convert_model() <openvino_docs_MO_DG_prepare_model_convert_model_MO_OVC_transition>`

@endsphinxdirective

