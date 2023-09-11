# Model Preparation {#openvino_docs_model_processing_introduction}

@sphinxdirective

.. meta::
   :description: Preparing models for OpenVINO Runtime. Learn about the methods
                 used to read, convert and compile models from different frameworks.

.. toctree::
   :maxdepth: 1
   :hidden:

   Supported_Model_Formats
   openvino_docs_OV_Converter_UG_Conversion_Options
   openvino_docs_OV_Converter_UG_prepare_model_convert_model_Converting_Model
   openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition


Every deep learning workflow begins with obtaining a model. You can choose to prepare a custom one, use a ready-made solution and adjust it to your needs, or even download and run a pre-trained network from an online database, such as `TensorFlow Hub <https://tfhub.dev/>`__, `Hugging Face <https://huggingface.co/>`__, or `Torchvision models <https://pytorch.org/hub/>`__.

OpenVINO™ :doc:`supports several model representations <Supported_Model_Formats>` and allows converting them to it's own representation, `openvino.Model <api/ie_python_api/_autosummary/openvino.Model.html>`__ (`ov.Model <api/ie_python_api/_autosummary/openvino.runtime.Model.html>`__), providing a conversion API to this task. Converted model can be used for inference using one or multiple OpenVINO Hardware plugins. This chapter describes two variants of using conversion API: using a Python program or calling ``ovc`` command line tool.

.. note::

   Prior OpenVINO 2023.1 release, model conversion API was exposed as ``openvino.tools.mo.convert_model`` function and ``mo`` command line tool.
   Starting from 2023.1 release, a new simplified API was introduced: ``openvino.convert_model`` function and ``ovc`` command line tool as a replacement for ``openvino.tools.mo.convert_model```
   and ``mo`` correspondingly, which are considered to be legacy now. All new users are recommended to use these new methods instead of the old methods. Please note that the new API and old API do not
   provide the same level of features, that means the new tools are not always backward compatible with the old ones. Please consult with :doc:`Model Conversion API Transition Guide <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`.

Convert a model in Python: ``convert_model``
############################################

Model conversion API is exposed in Python by means of ``openvino.convert_model`` function which converts a model from original framework representation, for example Pytorch or TensorFlow, to the object of type ``openvino.Model``. The resulting ``openvino.Model`` can be inferred in the same application (Python script or Jupiter Notebook) or saved into a file with ``openvino.save_model`` for later use. There are several examples of using ``openvino.convert_model`` below based on popular public model repositories:

.. tab-set::

   .. tab-item:: Torchvision

      .. code-block:: py
         :force:

            import torch
            from torchvision.models import resnet50
            import openvino as ov

            model = resnet50(pretrained=True)

            # prepare input_data
            input_data = torch.rand(1, 3, 224, 224)

            ov_model = ov.convert_model(model, example_input=input_data)

            ###### Option 1: Save to OpenVINO IR:

            # save model to OpenVINO IR for later use
            ov.save_model(ov_model, 'model.xml')

            ###### Option 2: Compile and infer with OpenVINO:

            # compile model
            compiled_model = ov.compile_model(ov_model)

            # run the inference
            result = compiled_model(input_data)

   .. tab-item:: HuggingFace Transformers

      .. code-block:: py

         from transformers import BertTokenizer, BertModel

         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
         model = BertModel.from_pretrained("bert-base-uncased")
         text = "Replace me by any text you'd like."
         encoded_input = tokenizer(text, return_tensors='pt')

         import openvino as ov
         ov_model = ov.convert_model(model, example_input={**encoded_input})

         ###### Option 1: Save to OpenVINO IR:

         # save model to OpenVINO IR for later use
         ov.save_model(ov_model, 'model.xml')

         ###### Option 2: Compile and infer with OpenVINO:

         # compile model
         compiled_model = ov.compile_model(ov_model)

         # prepare input_data your way using HF tokenizer or your own tokenizer
         # encoded_input is reused here for simplicity

         # run the inference
         result = compiled_model({**encoded_input})

   .. tab-item:: Keras Applications

      .. code-block:: py

         import tensorflow as tf
         import openvino as ov

         tf_model = tf.keras.applications.ResNet50(weights="imagenet")
         ov_model = ov.convert_model(tf_model)

         ###### Option 1: Save to OpenVINO IR:

         # save model to OpenVINO IR for later use
         ov.save_model(ov_model, 'model.xml')

         ###### Option 2: Compile and infer with OpenVINO:

         # compile model
         compiled_model = ov.compile_model(ov_model)

         # prepare input_data your way
         import numpy as np
         input_data = np.random.rand(1, 224, 224, 3)

         # run the inference
         result = compiled_model(input_data)

   .. tab-item:: TensorFlow Hub

      .. code-block:: py

         import tensorflow as tf
         import tensorflow_hub as hub
         import openvino as ov

         model = tf.keras.Sequential([
               hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5")
         ])

         # Check model page for information about input shape: https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5
         model.build([None, 224, 224, 3])

         model.save('mobilenet_v1_100_224')  # use a temporary directory
         ov_model = ov.convert_model('mobilenet_v1_100_224')

         ###### Option 1: Save to OpenVINO IR:

         ov.save_model(ov_model, 'model.xml')

         ###### Option 2: Compile and infer with OpenVINO:

         compiled_model = ov.compile_model(ov_model)

         # prepare input_data your way
         import numpy as np
         input_data = np.random.rand(1, 224, 224, 3)

         # run the inference
         result = compiled_model(input_data)

   .. tab-item:: ONNX Model Hub

      .. code-block:: py

         import onnx

         model = onnx.hub.load("resnet50")
         onnx.save(model, 'resnet50.onnx')  # use a temporary file for model

         import openvino as ov
         ov_model = ov.convert_model('resnet50.onnx')

         ###### Option 1: Save to OpenVINO IR:

         # save model to OpenVINO IR for later use
         ov.save_model(ov_model, 'model.xml')

         ###### Option 2: Compile and infer with OpenVINO:

         # compile model
         compiled_model = ov.compile_model(ov_model)

         # prepare input_data your way
         import numpy as np
         input_data = np.random.rand(1, 3, 224, 224)

         # run the inference
         result = compiled_model(input_data)

In Option 1, where ``openvino.save_model`` function is used, an OpenVINO model will be serialized in the file system as a pair of files with extensions ``.xml`` and ``.bin``. This pair of files is called OpenVINO Intermediate Representation format (OpenVINO IR, or just IR) and useful for efficient model deployment. OpenVINO IR is intended to be loaded in another application with the ``openvino.Core.read_model`` for inference, please see details in :doc:`OpenVINO™ Runtime documentation <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.

Option 2, where ``openvino.compile_model`` is used, provides a convenient way to quickly switch from framework-based code to OpenVINO-based code in your already existing Python inference application. In this case, the converted model isn't saved to the IR, instead, the model is compiled and used for inference in the same application.

Following option 1 means separating the model conversion and the model inference into two different applications. This approach addresses deployment scenarios where it is required to minimize extra dependencies and speed up model loading in the end inference application. For example, to convert a PyTorch model to OpenVINO, ``torch`` Python module is required as a dependency and the conversion must be performed in Python. It takes extra time and memory that wouldn't be required for inference of the converted model. But when the converted model is saved into the IR files with ``openvino.save_model``, it can be loaded in a separate application without ``torch`` dependency and without the need to spend time for model conversion. Also, the inference application can be written in other programming languages supported by OpenVINO, for example, in C++, and Python is not required to be installed for the inference application in this case.

Before saving the model to OpenVINO IR, consider applying :doc:`Post-training Optimization <ptq_introduction>` to enable more efficient inference and smaller model size.

The figure below illustrates the typical workflow for deploying a trained deep-learning model.

**TODO: Update BASIC_FLOW_MO_simplified.svg and replace 'mo' with 'ovc'**
.. image:: ./_static/images/model_conversion_diagram.svg
   :alt: model conversion diagram

Convert a model in CLI: ``ovc``
###############################

Another option to convert a model is to use ``ovc`` command-line tool. ``ovc`` stands for OpenVINO Model Converter and combines both ``openvino.convert_model`` and ``openvino.save_model`` together. It is convenient to use if the original model is ready for inference and represented in a file of one of the supported formats: ONNX, TensorFlow, TensorFlow Lite, or PaddlePaddle. As the result of the conversion, ``ovc`` produces OpenVINO IR as a pair of ``.xml`` and ``.bin`` files, which needs to be read with the ``ov.read_model()`` method. Then, you can compile and infer the ``ov.Model`` later with :doc:`OpenVINO™ Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`

.. note::
   PyTorch models cannot be converted with ``ovc``, use ``openvino.convert_model`` instead.

The results of both ``ovc`` and ``openvino.convert_model``/``openvino.save_model`` conversion methods described above are the same. You can choose one of them, depending on what is most convenient for you. Keep in mind that there should not be any differences in the results of model conversion if the same set of parameters is used and the model is saved into OpenVINO IR.

Cases When Model Preparation is Not Required
############################################

If a model is represented as a single file from ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite (check :doc:`TensorFlow Frontend Capabilities and Limitations <openvino_docs_MO_DG_TensorFlow_Frontend>`), then it doesn't require a separate step for model conversion and saving to IR, that is ``openvino.convert_model`` and ``openvino.save_model``, or ``ovc``. OpenVINO provides C++ and Python APIs for reading such models by just calling the ``openvino.Core.read_model`` or ``openvino.Core.compile_model`` methods. These methods perform conversion of the model from the original representation. Besides this conversion takes some extra time in comparison to reading the same model from prepared OpenVINO IR, it may be convenient in cases when it is required to read a model in the original format in C++ as ``openvino.convert_model`` is available in Python only. Preparing OpenVINO IR as a dedicated step and then using this IR in an application dedicated to inference is still the recommended way for the efficient model deployment for OpenVINO runtime.

Additional Resources
####################

The following articles describe in details how to obtain and prepare your model depending on the source model type:

* :doc:`See the supported formats and how to use them in your project <Supported_Model_Formats>`.
* :doc:`Convert different model formats to the ov.Model format <openvino_docs_OV_Converter_UG_Deep_Learning_Model_Optimizer_DevGuide>`.

To achieve the best model inference performance and more compact OpenVINO IR representation follow:

* :doc:`Post-training optimization <ptq_introduction>`
* :doc:`Model inference in OpenVINO Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`

If you are using legacy conversion API (``mo`` or ``openvino.tools.mo.convert_model``), please refer to the following materials:

* :doc:`Transition from legacy mo and ov.tools.mo.convert_model <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`
* :doc:`Legacy Model Conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`

@endsphinxdirective
