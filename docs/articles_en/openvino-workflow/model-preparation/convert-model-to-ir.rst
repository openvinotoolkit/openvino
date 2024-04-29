.. {#openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_IR}


Convert to OpenVINO IR
=============================================

.. meta::
   :description: Convert models from the original framework to OpenVINO representation.

.. toctree::
   :maxdepth: 1
   :hidden:

   Convert from PyTorch <convert-model-pytorch>
   Convert from TensorFlow <convert-model-tensorflow>
   Convert from ONNX <convert-model-onnx>
   Convert from TensorFlow Lite <convert-model-tensorflow-lite>
   Convert from PaddlePaddle <convert-model-paddle>


:doc:`IR (Intermediate Representation) <../../documentation/openvino-ir-format>` is OpenVINO own format consisting of  ``.xml`` and ``.bin`` files.
Convert the model into OpenVINO IR for `better performance <#ir-conversion-benefits>`__.

Convert Models
##############################################

Here are code examples of how to use these methods with different model formats:

.. tab-set::

   .. tab-item:: PyTorch
      :sync: torch

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            * The ``convert_model()`` method:

              This is the only method applicable to PyTorch models.

              .. dropdown:: List of supported formats:

                 * **Python objects**:

                   * ``torch.nn.Module``
                   * ``torch.jit.ScriptModule``
                   * ``torch.jit.ScriptFunction``
                   * ``torch.export.ExportedProgram``

              .. code-block:: py
                 :force:

                 import torchvision
                 import openvino as ov

                 model = torchvision.models.resnet50(weights='DEFAULT')
                 ov_model = ov.convert_model(model)
                 compiled_model = ov.compile_model(ov_model, "AUTO")

              For more details on conversion, refer to the
              :doc:`guide <convert-model-pytorch>`
              and an example `tutorial <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-onnx-to-openvino.ipynb>`__
              on this topic.

   .. tab-item:: TensorFlow
      :sync: tf

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            * The ``convert_model()`` method:

              When you use the ``convert_model()`` method, you have more control and you can specify additional adjustments for ``ov.Model``. The ``read_model()`` and ``compile_model()`` methods are easier to use, however, they do not have such capabilities. With ``ov.Model`` you can choose to optimize, compile and run inference on it or serialize it into a file for subsequent use.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * SavedModel - ``<SAVED_MODEL_DIRECTORY>`` or ``<INPUT_MODEL>.pb``
                   * Checkpoint - ``<INFERENCE_GRAPH>.pb`` or ``<INFERENCE_GRAPH>.pbtxt``
                   * MetaGraph - ``<INPUT_META_GRAPH>.meta``

                 * **Python objects**:

                   * ``tf.keras.Model``
                   * ``tf.keras.layers.Layer``
                   * ``tf.Module``
                   * ``tf.compat.v1.Graph``
                   * ``tf.compat.v1.GraphDef``
                   * ``tf.function``
                   * ``tf.compat.v1.session``
                   * ``tf.train.checkpoint``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model("saved_model.pb")
                 compiled_model = ov.compile_model(ov_model, "AUTO")

              For more details on conversion, refer to the
              :doc:`guide <convert-model-tensorflow>`
              and an example `tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-classification-to-openvino>`__
              on this topic.

            * The ``read_model()`` and ``compile_model()`` methods:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * SavedModel - ``<SAVED_MODEL_DIRECTORY>`` or ``<INPUT_MODEL>.pb``
                   * Checkpoint - ``<INFERENCE_GRAPH>.pb`` or ``<INFERENCE_GRAPH>.pbtxt``
                   * MetaGraph - ``<INPUT_META_GRAPH>.meta``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 core = ov.Core()
                 ov_model = core.read_model("saved_model.pb")
                 compiled_model = ov.compile_model(ov_model, "AUTO")

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: C++
            :sync: cpp

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * SavedModel - ``<SAVED_MODEL_DIRECTORY>`` or ``<INPUT_MODEL>.pb``
                   * Checkpoint - ``<INFERENCE_GRAPH>.pb`` or ``<INFERENCE_GRAPH>.pbtxt``
                   * MetaGraph - ``<INPUT_META_GRAPH>.meta``

              .. code-block:: cpp

                 ov::CompiledModel compiled_model = core.compile_model("saved_model.pb", "AUTO");

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: C
            :sync: c

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * SavedModel - ``<SAVED_MODEL_DIRECTORY>`` or ``<INPUT_MODEL>.pb``
                   * Checkpoint - ``<INFERENCE_GRAPH>.pb`` or ``<INFERENCE_GRAPH>.pbtxt``
                   * MetaGraph - ``<INPUT_META_GRAPH>.meta``

              .. code-block:: c

                 ov_compiled_model_t* compiled_model = NULL;
                 ov_core_compile_model_from_file(core, "saved_model.pb", "AUTO", 0, &compiled_model);

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: CLI
            :sync: cli

            You can use ``ovc`` command-line tool to convert a model to IR. The obtained IR can then be read by ``read_model()`` and inferred.

            .. code-block:: sh

               ovc <INPUT_MODEL>.pb

            For details on the conversion, refer to the
            :doc:`article <../model-preparation>`.

   .. tab-item:: TensorFlow Lite
      :sync: tflite

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            * The ``convert_model()`` method:

              When you use the ``convert_model()`` method, you have more control and you can specify additional adjustments for ``ov.Model``. The ``read_model()`` and ``compile_model()`` methods are easier to use, however, they do not have such capabilities. With ``ov.Model`` you can choose to optimize, compile and run inference on it or serialize it into a file for subsequent use.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.tflite``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model("<INPUT_MODEL>.tflite")
                 compiled_model = ov.compile_model(ov_model, "AUTO")

              For more details on conversion, refer to the
              :doc:`guide <convert-model-tensorflow-lite>`
              and an example `tutorial <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/tflite-to-openvino/tflite-to-openvino.ipynb>`__
              on this topic.


            * The ``read_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.tflite``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 core = ov.Core()
                 ov_model = core.read_model("<INPUT_MODEL>.tflite")
                 compiled_model = ov.compile_model(ov_model, "AUTO")

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.tflite``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 compiled_model = ov.compile_model("<INPUT_MODEL>.tflite", "AUTO")

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.


         .. tab-item:: C++
            :sync: cpp

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.tflite``

              .. code-block:: cpp

                 ov::CompiledModel compiled_model = core.compile_model("<INPUT_MODEL>.tflite", "AUTO");

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: C
            :sync: c

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.tflite``

              .. code-block:: c

                 ov_compiled_model_t* compiled_model = NULL;
                 ov_core_compile_model_from_file(core, "<INPUT_MODEL>.tflite", "AUTO", 0, &compiled_model);

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: CLI
            :sync: cli

            * The ``convert_model()`` method:

              You can use ``mo`` command-line tool to convert a model to IR. The obtained IR can then be read by ``read_model()`` and inferred.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.tflite``

              .. code-block:: sh

                 ovc <INPUT_MODEL>.tflite

              For details on the conversion, refer to the
              :doc:`article <convert-model-tensorflow-lite>`.

   .. tab-item:: ONNX
      :sync: onnx

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            * The ``convert_model()`` method:

              When you use the ``convert_model()`` method, you have more control and you can specify additional adjustments for ``ov.Model``. The ``read_model()`` and ``compile_model()`` methods are easier to use, however, they do not have such capabilities. With ``ov.Model`` you can choose to optimize, compile and run inference on it or serialize it into a file for subsequent use.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model("<INPUT_MODEL>.onnx")
                 compiled_model = ov.compile_model(ov_model, "AUTO")

              For more details on conversion, refer to the
              :doc:`guide <convert-model-onnx>`
              and an example `tutorial <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-onnx-to-openvino.ipynb>`__
              on this topic.


            * The ``read_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 core = ov.Core()
                 ov_model = core.read_model("<INPUT_MODEL>.onnx")
                 compiled_model = ov.compile_model(ov_model, "AUTO")

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 compiled_model = ov.compile_model("<INPUT_MODEL>.onnx", "AUTO")

              For a guide on how to run inference, see how to :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.


         .. tab-item:: C++
            :sync: cpp

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: cpp

                 ov::CompiledModel compiled_model = core.compile_model("<INPUT_MODEL>.onnx", "AUTO");

              For a guide on how to run inference, see how to :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: C
            :sync: c

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: c

                 ov_compiled_model_t* compiled_model = NULL;
                 ov_core_compile_model_from_file(core, "<INPUT_MODEL>.onnx", "AUTO", 0, &compiled_model);

              For details on the conversion, refer to the :doc:`article <convert-model-onnx>`

         .. tab-item:: CLI
            :sync: cli

            * The ``convert_model()`` method:

              You can use ``mo`` command-line tool to convert a model to IR. The obtained IR can then be read by ``read_model()`` and inferred.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: sh

                 ovc <INPUT_MODEL>.onnx

              For details on the conversion, refer to the
              :doc:`article <convert-model-onnx>`

   .. tab-item:: PaddlePaddle
      :sync: pdpd

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            * The ``convert_model()`` method:

              When you use the ``convert_model()`` method, you have more control and you can specify additional adjustments for ``ov.Model``. The ``read_model()`` and ``compile_model()`` methods are easier to use, however, they do not have such capabilities. With ``ov.Model`` you can choose to optimize, compile and run inference on it or serialize it into a file for subsequent use.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

                 * **Python objects**:

                   * ``paddle.hapi.model.Model``
                   * ``paddle.fluid.dygraph.layers.Layer``
                   * ``paddle.fluid.executor.Executor``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 ov_model = ov.convert_model("<INPUT_MODEL>.pdmodel")
                 compiled_model = ov.compile_model(ov_model, "AUTO")

              For more details on conversion, refer to the
              :doc:`guide <convert-model-paddle>`
              and an example `tutorial <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/paddle-to-openvino/paddle-to-openvino-classification.ipynb>`__
              on this topic.

            * The ``read_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 core = ov.Core()
                 ov_model = core.read_model("<INPUT_MODEL>.pdmodel")
                 compiled_model = ov.compile_model(ov_model, "AUTO")

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

              .. code-block:: py
                 :force:

                 import openvino as ov

                 compiled_model = ov.compile_model("<INPUT_MODEL>.pdmodel", "AUTO")

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: C++
            :sync: cpp

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

              .. code-block:: cpp

                 ov::CompiledModel compiled_model = core.compile_model("<INPUT_MODEL>.pdmodel", "AUTO");

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: C
            :sync: c

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

              .. code-block:: c

                 ov_compiled_model_t* compiled_model = NULL;
                 ov_core_compile_model_from_file(core, "<INPUT_MODEL>.pdmodel", "AUTO", 0, &compiled_model);

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: CLI
            :sync: cli

            * The ``convert_model()`` method:

              You can use ``mo`` command-line tool to convert a model to IR. The obtained IR can then be read by ``read_model()`` and inferred.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

              .. code-block:: sh

                 ovc <INPUT_MODEL>.pdmodel

              For details on the conversion, refer to the
              :doc:`article <convert-model-paddle>`.


* :doc:`How to convert PyTorch <convert-model-pytorch>`
* :doc:`How to convert ONNX <convert-model-onnx>`
* :doc:`How to convert TensorFlow <convert-model-tensorflow>`
* :doc:`How to convert TensorFlow Lite <convert-model-tensorflow-lite>`
* :doc:`How to convert PaddlePaddle <convert-model-paddle>`

To choose the best workflow for your application, read the :doc:`Model Preparation section <../model-preparation>`.

Refer to the list of all supported conversion options in :doc:`Conversion Parameters <conversion-parameters>`.

IR Conversion Benefits
################################################


| **Saving to IR to improve first inference latency**
|    When first inference latency matters, rather than convert the framework model each time it is loaded, which may take some time depending on its size, it is better to do it once. Save the model as an OpenVINO IR with ``save_model`` and then load it with ``read_model`` as needed. This should improve the time it takes the model to make the first inference as it avoids the conversion step.

| **Saving to IR in FP16 to save space**
|    Save storage space, even more so if FP16 is used as it may cut the size by about 50%, especially useful for large models, like Llama2-7B.

| **Saving to IR to avoid large dependencies in inference code**
|    Frameworks such as TensorFlow and PyTorch tend to be large dependencies (multiple gigabytes), and not all inference environments have enough space to hold them.
|    Converting models to OpenVINO IR allows them to be used in an environment where OpenVINO is the only dependency, so much less disk space is needed.
|    Loading and compiling with OpenVINO directly usually takes less runtime memory than loading the model in the source framework and then converting and compiling it.

An example showing how to take advantage of OpenVINO IR, saving a model in OpenVINO IR once, using it many times, is shown below:

.. code-block:: py

   # Run once

   import openvino as ov
   import tensorflow as tf

   # 1. Convert model created with TF code
   model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")
   ov_model = ov.convert_model(model)

   # 2. Save model as OpenVINO IR
   ov.save_model(ov_model, 'model.xml', compress_to_fp16=True) # enabled by default

   # Repeat as needed

   import openvino as ov

   # 3. Load model from file
   core = ov.Core()
   ov_model = core.read_model("model.xml")

   # 4. Compile model from memory
   compiled_model = ov.compile_model(ov_model)

Additional Resources
####################

* :doc:`Transition guide from the legacy to new conversion API <../../documentation/legacy-features/transition-legacy-conversion-api>`

