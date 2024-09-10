[LEGACY] Supported Model Formats
=====================================

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Supported Model Formats <../../../../openvino-workflow/model-preparation>` article.

.. toctree::
   :maxdepth: 1
   :hidden:

   Converting a TensorFlow Model <[legacy]-supported-model-formats/[legacy]-convert-tensorflow>
   Converting an ONNX Model <[legacy]-supported-model-formats/[legacy]-convert-onnx>
   Converting a PyTorch Model <[legacy]-supported-model-formats/[legacy]-convert-pytorch>
   Converting a TensorFlow Lite Model <[legacy]-supported-model-formats/[legacy]-convert-tensorflow-lite>
   Converting a PaddlePaddle Model <[legacy]-supported-model-formats/[legacy]-convert-paddle>
   Model Conversion Tutorials <[legacy]-supported-model-formats/[legacy]-conversion-tutorials>

.. meta::
   :description: Learn about supported model formats and the methods used to convert, read, and compile them in OpenVINO™.


**OpenVINO IR (Intermediate Representation)** - the proprietary and default format of OpenVINO, benefiting from the full extent of its features. All other supported model formats, as listed below, are converted to :doc:`OpenVINO IR <../../../openvino-ir-format>` to enable inference. Consider storing your model in this format to minimize first-inference latency, perform model optimization, and, in some cases, save space on your drive.

**PyTorch, TensorFlow, ONNX, and PaddlePaddle** - can be used with OpenVINO Runtime API directly,
which means you do not need to save them as OpenVINO IR before including them in your application.
OpenVINO can read, compile, and convert them automatically, as part of its pipeline.

In the Python API, these options are provided as three separate methods:
``read_model()``, ``compile_model()``, and ``convert_model()``.
The ``convert_model()`` method enables you to perform additional adjustments
to the model, such as setting shapes, changing model input types or layouts,
cutting parts of the model, freezing inputs, etc. For a detailed description
of the conversion process, see the
:doc:`model conversion guide <../legacy-conversion-api>`.

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

              .. code-block:: py
                 :force:

                 import openvino
                 import torchvision
                 from openvino.tools.mo import convert_model
                 core = openvino.Core()

                 model = torchvision.models.resnet50(weights='DEFAULT')
                 ov_model = convert_model(model)
                 compiled_model = core.compile_model(ov_model, "AUTO")

              For more details on conversion, refer to the
              :doc:`guide <[legacy]-supported-model-formats/[legacy]-convert-pytorch>`
              and an example `tutorial <https://docs.openvino.ai/2024/notebooks/pytorch-onnx-to-openvino-with-output.html>`__
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

                 import openvino
                 from openvino.tools.mo import convert_model

                 core = openvino.Core()
                 ov_model = convert_model("saved_model.pb")
                 compiled_model = core.compile_model(ov_model, "AUTO")

              For more details on conversion, refer to the
              :doc:`guide <[legacy]-supported-model-formats/[legacy]-convert-tensorflow>`
              and an example `tutorial <https://docs.openvino.ai/2024/notebooks/tensorflow-to-openvino-with-output.html>`__
              on this topic.

            * The ``read_model()`` and ``compile_model()`` methods:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * SavedModel - ``<SAVED_MODEL_DIRECTORY>`` or ``<INPUT_MODEL>.pb``
                   * Checkpoint - ``<INFERENCE_GRAPH>.pb`` or ``<INFERENCE_GRAPH>.pbtxt``
                   * MetaGraph - ``<INPUT_META_GRAPH>.meta``

              .. code-block:: py
                 :force:

                 ov_model = read_model("saved_model.pb")
                 compiled_model = core.compile_model(ov_model, "AUTO")

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.

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
              :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.

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
              :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: CLI
            :sync: cli

            You can use ``mo`` command-line tool to convert a model to IR. The obtained IR can then be read by ``read_model()`` and inferred.

            .. code-block:: sh

               mo --input_model <INPUT_MODEL>.pb

            For details on the conversion, refer to the
            :doc:`article <[legacy]-supported-model-formats/[legacy]-convert-tensorflow>`.

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

                 import openvino
                 from openvino.tools.mo import convert_model

                 core = openvino.Core()
                 ov_model = convert_model("<INPUT_MODEL>.tflite")
                 compiled_model = core.compile_model(ov_model, "AUTO")

              For more details on conversion, refer to the
              :doc:`guide <[legacy]-supported-model-formats/[legacy]-convert-tensorflow>`
              and an example `tutorial <https://docs.openvino.ai/2024/notebooks/tflite-to-openvino-with-output.html>`__
              on this topic.


            * The ``read_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.tflite``

              .. code-block:: py
                 :force:

                 import openvino

                 core = openvino.Core()
                 ov_model = core.read_model("<INPUT_MODEL>.tflite")
                 compiled_model = core.compile_model(ov_model, "AUTO")

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.tflite``

              .. code-block:: py
                 :force:

                 import openvino

                 core = openvino.Core()
                 compiled_model = core.compile_model("<INPUT_MODEL>.tflite", "AUTO")

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.


         .. tab-item:: C++
            :sync: cpp

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.tflite``

              .. code-block:: cpp

                 ov::CompiledModel compiled_model = core.compile_model("<INPUT_MODEL>.tflite", "AUTO");

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.

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
              :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: CLI
            :sync: cli

            * The ``convert_model()`` method:

              You can use ``mo`` command-line tool to convert a model to IR. The obtained IR can then be read by ``read_model()`` and inferred.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.tflite``

              .. code-block:: sh

                 mo --input_model <INPUT_MODEL>.tflite

              For details on the conversion, refer to the
              :doc:`article <[legacy]-supported-model-formats/[legacy]-convert-tensorflow-lite>`.

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

                 import openvino
                 from openvino.tools.mo import convert_model

                 core = openvino.Core()
                 ov_model = convert_model("<INPUT_MODEL>.onnx")
                 compiled_model = core.compile_model(ov_model, "AUTO")

              For more details on conversion, refer to the
              :doc:`guide <[legacy]-supported-model-formats/[legacy]-convert-onnx>`
              and an example `tutorial <https://docs.openvino.ai/2024/notebooks/pytorch-onnx-to-openvino-with-output.html>`__
              on this topic.


            * The ``read_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: py
                 :force:

                 import openvino
                 core = openvino.Core()

                 ov_model = core.read_model("<INPUT_MODEL>.onnx")
                 compiled_model = core.compile_model(ov_model, "AUTO")

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: py
                 :force:

                 import openvino
                 core = openvino.Core()

                 compiled_model = core.compile_model("<INPUT_MODEL>.onnx", "AUTO")

              For a guide on how to run inference, see how to :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.


         .. tab-item:: C++
            :sync: cpp

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: cpp

                 ov::CompiledModel compiled_model = core.compile_model("<INPUT_MODEL>.onnx", "AUTO");

              For a guide on how to run inference, see how to :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: C
            :sync: c

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: c

                 ov_compiled_model_t* compiled_model = NULL;
                 ov_core_compile_model_from_file(core, "<INPUT_MODEL>.onnx", "AUTO", 0, &compiled_model);

              For details on the conversion, refer to the :doc:`article <[legacy]-supported-model-formats/[legacy]-convert-onnx>`

         .. tab-item:: CLI
            :sync: cli

            * The ``convert_model()`` method:

              You can use ``mo`` command-line tool to convert a model to IR. The obtained IR can then be read by ``read_model()`` and inferred.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.onnx``

              .. code-block:: sh

                 mo --input_model <INPUT_MODEL>.onnx

              For details on the conversion, refer to the
              :doc:`article <[legacy]-supported-model-formats/[legacy]-convert-onnx>`

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

                 import openvino
                 from openvino.tools.mo import convert_model

                 core = openvino.Core()
                 ov_model = convert_model("<INPUT_MODEL>.pdmodel")
                 compiled_model = core.compile_model(ov_model, "AUTO")

              For more details on conversion, refer to the
              :doc:`guide <[legacy]-supported-model-formats/[legacy]-convert-paddle>`
              and an example `tutorial <https://docs.openvino.ai/2024/notebooks/paddle-to-openvino-classification-with-output.html>`__
              on this topic.

            * The ``read_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

              .. code-block:: py
                 :force:

                 import openvino
                 core = openvino.Core()

                 ov_model = read_model("<INPUT_MODEL>.pdmodel")
                 compiled_model = core.compile_model(ov_model, "AUTO")

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

              .. code-block:: py
                 :force:

                 import openvino
                 core = openvino.Core()

                 compiled_model = core.compile_model("<INPUT_MODEL>.pdmodel", "AUTO")

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: C++
            :sync: cpp

            * The ``compile_model()`` method:

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

              .. code-block:: cpp

                 ov::CompiledModel compiled_model = core.compile_model("<INPUT_MODEL>.pdmodel", "AUTO");

              For a guide on how to run inference, see how to
              :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.

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
              :doc:`Integrate OpenVINO™ with Your Application <../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`.

         .. tab-item:: CLI
            :sync: cli

            * The ``convert_model()`` method:

              You can use ``mo`` command-line tool to convert a model to IR. The obtained IR can then be read by ``read_model()`` and inferred.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

              .. code-block:: sh

                 mo --input_model <INPUT_MODEL>.pdmodel

              For details on the conversion, refer to the
              :doc:`article <[legacy]-supported-model-formats/[legacy]-convert-paddle>`.


As OpenVINO support for **MXNet, Caffe, and Kaldi formats** has been **discontinued**, converting these legacy formats
to OpenVINO IR or ONNX before running inference should be considered the default path for use with OpenVINO.

.. note::

   If you want to keep working with the legacy formats the old way, refer to a previous
   `OpenVINO LTS version and its documentation <https://docs.openvino.ai/2022.3/Supported_Model_Formats.html>`__ .

   OpenVINO versions of 2023 are mostly compatible with the old instructions,
   through a deprecated MO tool, installed with the deprecated OpenVINO Developer Tools package.

   `OpenVINO 2023.0 <https://docs.openvino.ai/archive/2023.0/Supported_Model_Formats.html>`__ is the last
   release officially supporting the MO conversion process for the legacy formats.


