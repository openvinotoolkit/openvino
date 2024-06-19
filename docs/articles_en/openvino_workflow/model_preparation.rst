.. {#openvino_docs_model_processing_introduction}

Model Preparation
=================


.. meta::
   :description: Preparing models for OpenVINO Runtime. Learn about the methods
                 used to read, convert and compile models from different frameworks.

.. toctree::
   :maxdepth: 1
   :hidden:

   Convert to OpenVINO Model <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_IR>
   Conversion Parameters <openvino_docs_OV_Converter_UG_Conversion_Options>
   Setting Input Shapes <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Converting_Model>


OpenVINO supports the following model formats:

* PyTorch,
* TensorFlow,
* TensorFlow Lite,
* ONNX,
* PaddlePaddle,
* OpenVINO IR.

The easiest way to obtain a model is to download it from an online database, such as `TensorFlow Hub <https://tfhub.dev/>`__, `Hugging Face <https://huggingface.co/>`__,
and `Torchvision models <https://pytorch.org/hub/>`__. Now you have two options:

* Skip model conversion and :doc:`run inference <openvino_docs_OV_UG_Integrate_OV_with_your_application>` directly from the **TensorFlow, TensorFlow Lite, ONNX, and PaddlePaddle** source format. Conversion
  will still be performed but it will happen automatically and "under the hood".
  This option, while convenient, offers lower performance and stability, as well as
  fewer optimization options.

* Explicitly :doc:`convert the model to OpenVINO IR <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_IR>`.
  This approach offers the best possible results and is the recommended one,
  especially for production-ready solutions. Consider storing your model in this format to minimize first-inference latency,
  perform model optimizations, and save space on your drive, in some cases. Explicit conversion can be done in two ways:

  * the `Python API functions <#convert-a-model-with-python-convert-model>`__ (``openvino.convert_model`` and ``openvino.save_model``)
  * the `ovc <#convert-a-model-in-cli-ovc>`__  command line tool.

  Once saved as :doc:`OpenVINO IR <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_IR>` (a set of ``.xml`` and ``.bin`` files),
  the model may be deployed with maximum performance. Because it is already optimized
  for :doc:`OpenVINO inference <openvino_docs_OV_UG_Integrate_OV_with_your_application>`, it can be read, compiled, and inferred with no additional delay.

.. note::

   Model conversion API prior to OpenVINO 2023.1 is considered deprecated.
   Existing and new projects are recommended to transition to the new
   solutions, keeping in mind that they are not fully backwards compatible
   with ``openvino.tools.mo.convert_model`` or the ``mo`` CLI tool.
   For more details, see the :doc:`Model Conversion API Transition Guide <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`.

   For PyTorch models, `Python API <#convert-a-model-with-python-convert-model>`__ is the only conversion option.

   TensorFlow may present additional considerations :doc:`TensorFlow Frontend Capabilities and Limitations <openvino_docs_MO_DG_TensorFlow_Frontend>`.

Model States
##############################################

There are three states a model in OpenVINO can be: saved on disk, loaded but not compiled (``ov.Model``) or loaded and compiled (``ov.CompiledModel``).

| **Saved on disk**
|    A model in this state consists of one or more files that fully represent the neural network. A model can be stored in different ways. For example:
|       OpenVINO IR: pair of .xml and .bin files
|       ONNX: .onnx file
|       TensorFlow: directory with a .pb file and two subfolders or just a .pb file
|       TensorFlow Lite: .tflite file
|       PaddlePaddle: .pdmodel file

| **Loaded but not compiled**
|    A model object (``ov.Model``) is created in memory either by parsing a file or converting an existing framework object. Inference cannot be done with this object yet as it is not attached to any specific device, but it allows customization such as reshaping its input, applying quantization or even adding preprocessing steps before compiling the model.

| **Loaded and compiled**
|   This state is achieved when one or more devices are specified for a model object to run on (``ov.CompiledModel``), allowing device optimizations to be made and enabling inference.

For more information on each function, see the :doc:`OpenVINO workflow <openvino_workflow>` page.

Convert a Model with Python: ``convert_model``
##############################################

The Model conversion API in Python uses the ``openvino.convert_model`` function,
turning a given model into the `openvino.Model <api/ie_python_api/_autosummary/openvino.runtime.Model.html>`__
object and loading it to memory. Now it can be: saved to a drive with ``openvino.save_model``
or further :doc:`optimized with NNCF <openvino_docs_model_optimization_guide>`
prior to saving.


See how to use ``openvino.convert_model`` with models from some of the most popular
public repositories:

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


* Saving the model, **Option 1**, is used as a separate step, outside of deployment.
  The file it provides is then used in the final software solution, resulting in
  maximum performance due to fewer dependencies and faster model loading.

* Compiling the model, **Option 2**, provides a convenient way to quickly switch from
  framework-based code to OpenVINO-based code in your existing Python inference application.
  The converted model is not saved to IR but compiled and used for inference within the same application.

Before saving the model to OpenVINO IR, consider :doc:`Post-training Optimization <ptq_introduction>`
to achieve more efficient inference and smaller model size.


Convert a Model in CLI: ``ovc``
###############################

``ovc`` is a command-line model converter, combining the ``openvino.convert_model``
and ``openvino.save_model`` functionalities, providing the exact same results, if the same set of
parameters is used for saving into OpenVINO IR. It converts files from one of the to :doc:`OpenVINO IR <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_IR>`, which can then be read, compiled,
and run by the final inference application.

.. note::
   PyTorch models cannot be converted with ``ovc``, use ``openvino.convert_model`` instead.

Additional Resources
####################

The following articles describe in detail how to obtain and prepare your model depending on the source model type:

* :doc:`Convert different model formats to the ov.Model format <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_IR>`.
* :doc:`Review all available conversion parameters <openvino_docs_OV_Converter_UG_Conversion_Options>`.

To achieve the best model inference performance and more compact OpenVINO IR representation follow:

* :doc:`Post-training optimization <ptq_introduction>`
* :doc:`Model inference in OpenVINO Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`

If you are still using the legacy conversion API (``mo`` or ``openvino.tools.mo.convert_model``), please refer to the following materials:

* :doc:`Transition from legacy mo and ov.tools.mo.convert_model <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`
* :doc:`Legacy Model Conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`




.. need to investigate python api article generation - api/ie_python_api/_autosummary/openvino.Model.html does not exist, api/ie_python_api/_autosummary/openvino.runtime.Model.html does.


