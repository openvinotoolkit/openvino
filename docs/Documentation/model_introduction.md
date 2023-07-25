# Model Preparation {#openvino_docs_model_processing_introduction}

@sphinxdirective

.. meta::
   :description: Preparing models for OpenVINO Runtime. Learn how to convert and compile models from different frameworks or read them directly.


.. toctree::
   :maxdepth: 1
   :hidden:

   Supported_Model_Formats
   openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide
   omz_tools_downloader


Every deep learning workflow begins with obtaining a model. You can choose to prepare a custom one, use a ready-made solution and adjust it to your needs, or even download and run a pre-trained network from an online database, such as `TensorFlow Hub <https://tfhub.dev/>`__, `Hugging Face <https://huggingface.co/>`__, `Torchvision models <https://pytorch.org/hub/>`__.

Import a model directly using ``read_model()``
#################################################

Model files (not Python objects) from :doc:`ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite <Supported_Model_Formats>`  (check :doc:`TensorFlow Frontend Capabilities and Limitations <openvino_docs_MO_DG_TensorFlow_Frontend>`) do not require a separate step for model conversion, that is ``mo.convert_model``. OpenVINO provides C++ and Python APIs for importing the models to OpenVINO Runtime directly by just calling the ``read_model`` method, therefore reducing application start time.

The ``read_model()`` method reads a model from a file and produces ``ov.Model``. If the file is in one of the supported original framework file :doc:`formats <Supported_Model_Formats>`, it is converted automatically to OpenVINO Intermediate Representation. If the file is already in the :doc:`OpenVINO IR format <openvino_ir>`, it is read "as-is", without any conversion involved.

You can still choose to convert a model from original framework to `openvino.runtime.Model <api/ie_python_api/_autosummary/openvino.runtime.Model.html>`__ (`ov.Model <api/ie_python_api/_autosummary/openvino.runtime.Model.html>`__ ), with a :doc:`tool <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>` dedicated to this task.

``ov.Model`` can be serialized to IR using the ``ov.serialize()`` method. The serialized IR can be further optimized using :doc:`Neural Network Compression Framework (NNCF) <ptq_introduction>` that applies post-training quantization methods.

.. note::

   Model conversion is still necessary in more complex cases, such as, new custom inputs/outputs in model pruning, adding pre-processing, or using Python conversion extensions.

Convert a model with Python using ``mo.convert_model()``
###########################################################

Model conversion API, specifically, the ``mo.convert_model()`` method converts a model from original framework to ``ov.Model``. ``mo.convert_model()`` returns ``ov.Model`` object in memory so the ``read_model()`` method is not required. The resulting ``ov.Model`` can be inferred in the same training environment (python script or Jupiter Notebook). ``mo.convert_model()`` provides a convenient way to quickly switch from framework-based code to OpenVINO-based code in your inference application. In addition to model files, ``mo.convert_model()`` can take OpenVINO extension objects constructed directly in Python for easier conversion of operations that are not supported in OpenVINO. The ``mo.convert_model()`` method also has a set of parameters to :doc:`cut the model <openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model>`, :doc:`set input shapes or layout <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`, :doc:`add preprocessing <openvino_docs_MO_DG_Additional_Optimization_Use_Cases>`, etc.

The figure below illustrates the typical workflow for deploying a trained deep learning model:

.. image:: _static/images/model_conversion_diagram.svg
   :alt: model conversion diagram


where IR is a pair of files describing the model:

* ``.xml`` - Describes the network topology.
* ``.bin`` - Contains the weights and biases binary data.

Convert a model using ``mo`` command-line tool
#################################################

Another option to convert a model is to use ``mo`` command-line tool. ``mo`` is a cross-platform tool that facilitates the transition between training and deployment environments, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices in the same measure, as the ``mo.convert_model`` method.

``mo`` requires the use of a pre-trained deep learning model in one of the supported formats: TensorFlow, TensorFlow Lite, PaddlePaddle, or ONNX. ``mo`` converts the model to the OpenVINO Intermediate Representation format (IR), which needs to be read with the ``ov.read_model()`` method. Then, you can compile and infer the ``ov.Model`` later with :doc:`OpenVINO™ Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.

The results of both ``mo`` and ``mo.convert_model()`` conversion methods described above are the same. You can choose one of them, depending on what is most convenient for you. Keep in mind that there should not be any differences in the results of model conversion if the same set of parameters is used.

This section describes how to obtain and prepare your model for work with OpenVINO to get the best inference results:

* :doc:`See the supported formats and how to use them in your project <Supported_Model_Formats>`.
* :doc:`Convert different model formats to the ov.Model format <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.


@endsphinxdirective
