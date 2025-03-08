Converting a PaddlePaddle Model
===============================


.. meta::
   :description: Learn how to convert a model from the
                 PaddlePaddle format to the OpenVINO Model.

This page provides general instructions on how to convert a model from the PaddlePaddle
format to the OpenVINO IR format using OpenVINO model conversion API. The instructions
are different depending on the PaddlePaddle model format.

.. note::

   PaddlePaddle model serialized in a file can be loaded by ``openvino.Core.read_model``
   or ``openvino.Core.compile_model`` methods by OpenVINO runtime API without preparing
   OpenVINO IR first. Refer to the
   :doc:`inference example <../running-inference>`
   for more details. Using ``openvino.convert_model`` is still recommended if model load
   latency matters for the inference application.

Converting PaddlePaddle Model Files
###################################

PaddlePaddle inference model includes ``.pdmodel`` (storing model structure) and
``.pdiparams`` (storing model weight). For details on how to export a PaddlePaddle
inference model, refer to the
`Exporting PaddlePaddle Inference Model <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/model_save_load_cn.html>`__
Chinese guide.

You can download a PaddlePaddle model from `Hugging Face <https://huggingface.co/models>`__.
To convert the model, use the ``ovc`` or ``openvino.convert_model``
and specify the path to the input ``.pdmodel`` model file:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py

         import openvino as ov
         ov.convert_model('your_model_file.pdmodel')

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc your_model_file.pdmodel

**For example**, this command converts a YOLOv3 PaddlePaddle model to OpenVINO IR model:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py

         import openvino as ov
         ov.convert_model('yolov3.pdmodel')

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc yolov3.pdmodel

Converting PaddlePaddle Python Model
####################################

Model conversion API supports passing PaddlePaddle models directly in Python without saving them to files in the user code.

Following PaddlePaddle model object types are supported:

* ``paddle.hapi.model.Model``
* ``paddle.fluid.dygraph.layers.Layer``
* ``paddle.fluid.executor.Executor``

Some PaddlePaddle models may require setting ``example_input`` or ``output`` for conversion as shown in the examples below:

* Example of converting ``paddle.hapi.model.Model`` format model:

  .. code-block:: py
     :force:

     import paddle
     import openvino as ov

     # create a paddle.hapi.model.Model format model
     resnet50 = paddle.vision.models.resnet50()
     x = paddle.static.InputSpec([1,3,224,224], 'float32', 'x')
     y = paddle.static.InputSpec([1,1000], 'float32', 'y')

     model = paddle.Model(resnet50, x, y)

     # convert to OpenVINO IR format
     ov_model = ov.convert_model(model)

     ov.save_model(ov_model, "resnet50.xml")

* Example of converting ``paddle.fluid.dygraph.layers.Layer`` format model:

  ``example_input`` is required while ``output`` is optional.  ``example_input`` accepts the following formats:

  ``list`` with tensor (``paddle.Tensor``) or InputSpec (``paddle.static.input.InputSpec``)

  .. code-block:: py
     :force:

     import paddle
     import openvino as ov

     # create a paddle.fluid.dygraph.layers.Layer format model
     model = paddle.vision.models.resnet50()
     x = paddle.rand([1,3,224,224])

     # convert to OpenVINO IR format
     ov_model = ov.convert_model(model, example_input=[x])

* Example of converting ``paddle.fluid.executor.Executor`` format model:

  ``example_input`` and ``output`` are required, which accept the following formats:

  ``list`` or ``tuple`` with variable(``paddle.static.data``)

  .. code-block:: py
     :force:

     import paddle
     import openvino as ov

     paddle.enable_static()

     # create a paddle.fluid.executor.Executor format model
     x = paddle.static.data(name="x", shape=[1,3,224])
     y = paddle.static.data(name="y", shape=[1,3,224])
     relu = paddle.nn.ReLU()
     sigmoid = paddle.nn.Sigmoid()
     y = sigmoid(relu(x))

     exe = paddle.static.Executor(paddle.CPUPlace())
     exe.run(paddle.static.default_startup_program())

     # convert to OpenVINO IR format
     ov_model = ov.convert_model(exe, example_input=[x], output=[y])

Supported PaddlePaddle Layers
#############################

For the list of supported standard layers, refer to the
:doc:`Supported Operations <../../documentation/compatibility-and-support/supported-operations>`
page.


Additional Resources
####################

Check out more examples of model conversion in
:doc:`interactive Python tutorials <../../get-started/learn-openvino/interactive-tutorials-python>`.

