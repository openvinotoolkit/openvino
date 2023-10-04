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


Every deep learning workflow begins with obtaining a model. You can choose to prepare
a custom one, use a ready-made solution and adjust it to your needs, or even download
and run a pre-trained network from an online database, such as 
`TensorFlow Hub <https://tfhub.dev/>`__, `Hugging Face <https://huggingface.co/>`__,
or `Torchvision models <https://pytorch.org/hub/>`__.

If your selected model is in one of the :doc:`OpenVINO™ supported model formats <Supported_Model_Formats>`,
you can use it directly, without the need to save as OpenVINO IR
(`openvino.Model <api/ie_python_api/_autosummary/openvino.Model.html>`__ -
`ov.Model <api/ie_python_api/_autosummary/openvino.Model.html>`__). 
For this purpose, you can use ``openvino.Core.read_model`` and ``openvino.Core.compile_model`` 
methods, so that conversion is performed automatically before inference, for 
maximum convenience. Note that for PyTorch models, Python API
is the only conversion option. TensorFlow may present additional considerations
:doc:`TensorFlow Frontend Capabilities and Limitations <openvino_docs_MO_DG_TensorFlow_Frontend>`.


For better performance and more optimization options, OpenVINO also offers a conversion
API with two possible approaches: the Python API functions (``openvino.convert_model`` 
and ``openvino.save_model``) and the ``ovc`` command line tool, which are described in detail in this article. 

.. note::
   
   Model conversion API prior to OpenVINO 2023.1 is considered deprecated. 
   Both existing and new projects are recommended to transition to the new
   solutions, keeping in mind that they are not fully backwards compatible 
   with ``openvino.tools.mo.convert_model`` or the ``mo`` CLI tool.
   For more details, see the :doc:`Model Conversion API Transition Guide <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`.




Convert a Model in Python: ``convert_model``
##############################################

You can use the Model conversion API in Python with the ``openvino.convert_model`` function. This function converts a model from its original framework representation, for example PyTorch or TensorFlow, to the object of type ``openvino.Model``. The resulting ``openvino.Model`` can be compiled with ``openvino.compile_model`` and inferred in the same application (Python script or Jupyter Notebook) or saved into a file using``openvino.save_model`` for future use. Below, there are examples of how to use the ``openvino.convert_model`` with models from popular public repositories:


.. tab-set::

   .. tab-item:: Torchvision

      .. code-block:: py
         :force:

            import openvino as ov
            import torch
            from torchvision.models import resnet50

            model = resnet50(weights='DEFAULT')

            # prepare input_data
            input_data = torch.rand(1, 3, 224, 224)

            ov_model = ov.convert_model(model, example_input=input_data)

            ###### Option 1: Save to OpenVINO IR:

            # save model to OpenVINO IR for later use
            ov.save_model(ov_model, 'model.xml')

            ###### Option 2: Compile and infer with OpenVINO:

            # compile model
            compiled_model = ov.compile_model(ov_model)

            # run inference
            result = compiled_model(input_data)

   .. tab-item:: Hugging Face Transformers

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

         # prepare input_data using HF tokenizer or your own tokenizer
         # encoded_input is reused here for simplicity

         # run inference
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

         # prepare input_data
         import numpy as np
         input_data = np.random.rand(1, 224, 224, 3)

         # run inference
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

         # prepare input_data
         import numpy as np
         input_data = np.random.rand(1, 224, 224, 3)

         # run inference
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

         # prepare input_data
         import numpy as np
         input_data = np.random.rand(1, 3, 224, 224)

         # run inference
         result = compiled_model(input_data)

In Option 1, where the ``openvino.save_model`` function is used, an OpenVINO model is serialized in the file system as two files with ``.xml`` and ``.bin`` extensions. This pair of files is called OpenVINO Intermediate Representation format (OpenVINO IR, or just IR) and useful for efficient model deployment. OpenVINO IR can be loaded into another application for inference using the ``openvino.Core.read_model`` function. For more details, refer to the :doc:`OpenVINO™ Runtime documentation <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.

Option 2, where ``openvino.compile_model`` is used, provides a convenient way to quickly switch from framework-based code to OpenVINO-based code in your existing Python inference application. In this case, the converted model is not saved to IR. Instead, the model is compiled and used for inference within the same application.

Option 1 separates model conversion and model inference into two different applications. This approach is useful for deployment scenarios requiring fewer extra dependencies and faster model loading in the end inference application. 

For example, converting a PyTorch model to OpenVINO usually demands the ``torch`` Python module and Python. This process can take extra time and memory. But, after the converted model is saved as OpenVINO IR with ``openvino.save_model``, it can be loaded in a separate application without requiring the ``torch`` dependency and the time-consuming conversion. The inference application can be written in other languages supported by OpenVINO, for example, in C++, and Python installation is not necessary for it to run.

Before saving the model to OpenVINO IR, consider applying :doc:`Post-training Optimization <ptq_introduction>` to enable more efficient inference and smaller model size.

The figure below illustrates the typical workflow for deploying a trained deep-learning model.

.. image:: ./_static/images/model_conversion_diagram.svg
   :alt: model conversion diagram

Convert a Model in CLI: ``ovc``
###############################

Another option for model conversion is to use ``ovc`` command-line tool, which stands for OpenVINO Model Converter. The tool combines both ``openvino.convert_model`` and ``openvino.save_model`` functionalities. It is convenient to use when the original model is ready for inference and is in one of the supported file formats: ONNX, TensorFlow, TensorFlow Lite, or PaddlePaddle. As a result, ``ovc`` produces an OpenVINO IR, consisting of ``.xml`` and ``.bin`` files, which needs to be read with the ``openvino.Core.read_model`` method. You can compile and infer the ``ov.Model`` later with :doc:`OpenVINO™ Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`

.. note::
   PyTorch models cannot be converted with ``ovc``, use ``openvino.convert_model`` instead.

The results of both ``ovc`` and ``openvino.convert_model``/``openvino.save_model`` conversion methods are the same. You can choose either of them based on your convenience. Note that there should not be any differences in the results of model conversion if the same set of parameters is used and the model is saved into OpenVINO IR.




Additional Resources
####################

The following articles describe in details how to obtain and prepare your model depending on the source model type:

* :doc:`Convert different model formats to the ov.Model format <Supported_Model_Formats>`.
* :doc:`Review all available conversion parameters <openvino_docs_OV_Converter_UG_Conversion_Options>`.

To achieve the best model inference performance and more compact OpenVINO IR representation follow:

* :doc:`Post-training optimization <ptq_introduction>`
* :doc:`Model inference in OpenVINO Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`

If you are using legacy conversion API (``mo`` or ``openvino.tools.mo.convert_model``), please refer to the following materials:

* :doc:`Transition from legacy mo and ov.tools.mo.convert_model <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`
* :doc:`Legacy Model Conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`




.. api/ie_python_api/_autosummary/openvino.Model.html is a broken link for some reason - need to investigate python api article generation


@endsphinxdirective
