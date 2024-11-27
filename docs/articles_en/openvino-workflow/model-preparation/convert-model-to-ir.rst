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
   Convert from JAX/Flax <convert-model-jax>



:doc:`OpenVINO IR <../../documentation/openvino-ir-format>` is the proprietary model format
used by OpenVINO, typically obtained by converting models of supported frameworks:

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

              When you use the ``convert_model()`` method, you have more control and you can
              specify additional adjustments for ``ov.Model``. The ``read_model()`` and
              ``compile_model()`` methods are easier to use, however, they do not have such
              capabilities. With ``ov.Model`` you can choose to optimize, compile and run
              inference on it or serialize it into a file for subsequent use.

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

            You can use ``ovc`` command-line tool to convert a model to IR. The obtained IR can
            then be read by ``read_model()`` and inferred.

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

              When you use the ``convert_model()`` method, you have more control and you can
              specify additional adjustments for ``ov.Model``. The ``read_model()`` and
              ``compile_model()`` methods are easier to use, however, they do not have such
              capabilities. With ``ov.Model`` you can choose to optimize, compile and run
              inference on it or serialize it into a file for subsequent use.

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

              You can use ``mo`` command-line tool to convert a model to IR. The obtained IR can
              then be read by ``read_model()`` and inferred.

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

              When you use the ``convert_model()`` method, you have more control and you can
              specify additional adjustments for ``ov.Model``. The ``read_model()`` and
              ``compile_model()`` methods are easier to use, however, they do not have such
              capabilities. With ``ov.Model`` you can choose to optimize, compile and run
              inference on it or serialize it into a file for subsequent use.

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

              You can use ``mo`` command-line tool to convert a model to IR. The obtained IR
              can then be read by ``read_model()`` and inferred.

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

              When you use the ``convert_model()`` method, you have more control and you can
              specify additional adjustments for ``ov.Model``. The ``read_model()`` and
              ``compile_model()`` methods are easier to use, however, they do not have such
              capabilities. With ``ov.Model`` you can choose to optimize, compile and run
              inference on it or serialize it into a file for subsequent use.

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

              You can use ``mo`` command-line tool to convert a model to IR. The obtained IR
              can then be read by ``read_model()`` and inferred.

              .. dropdown:: List of supported formats:

                 * **Files**:

                   * ``<INPUT_MODEL>.pdmodel``

              .. code-block:: sh

                 ovc <INPUT_MODEL>.pdmodel

              For details on the conversion, refer to the
              :doc:`article <convert-model-paddle>`.

   .. tab-item:: JAX/Flax
      :sync: torch

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            The ``convert_model()`` method is the only method applicable to JAX/Flax models.

            .. dropdown:: List of supported formats:

               * **Python objects**:

                 * ``jax._src.core.ClosedJaxpr``
                 * ``flax.linen.Module``

            * Conversion of the ``jax._src.core.ClosedJaxpr`` object

              .. code-block:: py
                 :force:

                 import jax
                 import jax.numpy as jnp
                 import openvino as ov

                 # let user have some JAX function
                 def jax_func(x, y):
                     return jax.lax.tanh(jax.lax.max(x, y))

                 # use example inputs for creation of ClosedJaxpr object
                 x = jnp.array([1.0, 2.0])
                 y = jnp.array([-1.0, 10.0])
                 jaxpr = jax.make_jaxpr(jax_func)(x, y)

                 ov_model = ov.convert_model(jaxpr)
                 compiled_model = ov.compile_model(ov_model, "AUTO")

            * Conversion of the ``flax.linen.Module`` object

              .. code-block:: py
                 :force:

                 import flax.linen as nn
                 import jax
                 import jax.numpy as jnp
                 import openvino as ov

                 # let user have some Flax module
                 class SimpleDenseModule(nn.Module):
                     features: int

                     @nn.compact
                     def __call__(self, x):
                         return nn.Dense(features=self.features)(x)

                 module = SimpleDenseModule(features=4)

                 # create example_input used in training
                 example_input = jnp.ones((2, 3))

                 # prepare parameters to initialize the module
                 # they can be also loaded from a disk
                 # using pickle, flax.serialization for deserialization
                 key = jax.random.PRNGKey(0)
                 params = module.init(key, example_input)
                 module = module.bind(params)

                 ov_model = ov.convert_model(module, example_input=example_input)
                 compiled_model = ov.compile_model(ov_model, "AUTO")

            For more details on conversion, refer to the :doc:`conversion guide <convert-model-jax>`.



These are basic examples, for detailed conversion instructions, see the individual guides on
:doc:`PyTorch <convert-model-pytorch>`, :doc:`ONNX <convert-model-onnx>`,
:doc:`TensorFlow <convert-model-tensorflow>`, :doc:`TensorFlow Lite <convert-model-tensorflow-lite>`,
:doc:`PaddlePaddle <convert-model-paddle>`, and :doc:`JAX/Flax <convert-model-jax>`.

Refer to the list of all supported conversion options in :doc:`Conversion Parameters <conversion-parameters>`.

IR Conversion Benefits
################################################

| **Saving to IR to improve first inference latency**
|    When first inference latency matters, rather than convert the framework model each time it
     is loaded, which may take some time depending on its size, it is better to do it once. Save
     the model as an OpenVINO IR with ``save_model`` and then load it with ``read_model`` as
     needed. This should improve the time it takes the model to make the first inference as it
     avoids the conversion step.

| **Saving to IR in FP16 to save space**
|    Save storage space, even more so if FP16 is used as it may cut the size by about 50%,
     especially useful for large models, like Llama2-7B.

| **Saving to IR to avoid large dependencies in inference code**
|    Frameworks such as TensorFlow, PyTorch, and JAX/Flax tend to be large dependencies for applications
     running inference (multiple gigabytes). Converting models to OpenVINO IR removes this
     dependency, as OpenVINO can run its inference with no additional components.
     This way, much less disk space is needed, while loading and compiling usually takes less
     runtime memory than loading the model in the source framework and then converting
     and compiling it.

Here is an example of how to benefit from OpenVINO IR, saving a model once and running it
multiple times:

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
* `Download models from Hugging Face <https://huggingface.co/models>`__.

