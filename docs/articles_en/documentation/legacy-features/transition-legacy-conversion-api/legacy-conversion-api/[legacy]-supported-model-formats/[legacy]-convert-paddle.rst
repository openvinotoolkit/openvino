[LEGACY] Converting a PaddlePaddle Model
======================================================


.. meta::
   :description: Learn how to convert a model from the
                 PaddlePaddle format to the OpenVINO Intermediate Representation.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Converting a PaddlePaddle Model <../../../../../openvino-workflow/model-preparation/convert-model-paddle>` article.


This page provides general instructions on how to convert a model from a PaddlePaddle format to the OpenVINO IR format using Model Optimizer. The instructions are different depending on PaddlePaddle model format.

.. note:: PaddlePaddle models are supported via FrontEnd API. You may skip conversion to IR and read models directly by OpenVINO runtime API. Refer to the :doc:`inference example <../../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>` for more details. Using ``convert_model`` is still necessary in more complex cases, such as new custom inputs/outputs in model pruning, adding pre-processing, or using Python conversion extensions.

Converting PaddlePaddle Model Inference Format
##############################################

PaddlePaddle inference model includes ``.pdmodel`` (storing model structure) and ``.pdiparams`` (storing model weight). For how to export PaddlePaddle inference model, please refer to the `Exporting PaddlePaddle Inference Model <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/model_save_load_cn.html>`__ Chinese guide.


To convert a PaddlePaddle model, use the ``mo`` script and specify the path to the input ``.pdmodel`` model file:

.. code-block:: sh

  mo --input_model <INPUT_MODEL>.pdmodel

**For example**, this command converts a yolo v3 PaddlePaddle network to OpenVINO IR network:

.. code-block:: sh

  mo --input_model=yolov3.pdmodel --input=image,im_shape,scale_factor --input_shape=[1,3,608,608],[1,2],[1,2] --reverse_input_channels --output=save_infer_model/scale_0.tmp_1,save_infer_model/scale_1.tmp_1

Converting PaddlePaddle Model From Memory Using Python API
##########################################################

Model conversion API supports passing the following PaddlePaddle models directly from memory:

* ``paddle.hapi.model.Model``
* ``paddle.fluid.dygraph.layers.Layer``
* ``paddle.fluid.executor.Executor``

When you convert certain PaddlePaddle models, you may need to set the ``example_input`` or ``example_output`` parameters first. Below you will find examples that show how to convert aforementioned model formats using the parameters.

* ``paddle.hapi.model.Model``

  .. code-block:: py
     :force:

     import paddle
     from openvino.tools.mo import convert_model

     # create a paddle.hapi.model.Model format model
     resnet50 = paddle.vision.models.resnet50()
     x = paddle.static.InputSpec([1,3,224,224], 'float32', 'x')
     y = paddle.static.InputSpec([1,1000], 'float32', 'y')

     model = paddle.Model(resnet50, x, y)

     # convert to OpenVINO IR format
     ov_model = convert_model(model)

     # optional: serialize OpenVINO IR to *.xml & *.bin
     from openvino.runtime import serialize
     serialize(ov_model, "ov_model.xml", "ov_model.bin")

* ``paddle.fluid.dygraph.layers.Layer``

  ``example_input`` is required while ``example_output`` is optional, and accept the following formats:

  ``list`` with tensor(``paddle.Tensor``) or InputSpec(``paddle.static.input.InputSpec``)

  .. code-block:: py
     :force:

     import paddle
     from openvino.tools.mo import convert_model

     # create a paddle.fluid.dygraph.layers.Layer format model
     model = paddle.vision.models.resnet50()
     x = paddle.rand([1,3,224,224])

     # convert to OpenVINO IR format
     ov_model = convert_model(model, example_input=[x])

* ``paddle.fluid.executor.Executor``

  ``example_input`` and ``example_output`` are required, and accept the following formats:

  ``list`` or ``tuple`` with variable(``paddle.static.data``)

  .. code-block:: py
     :force:

     import paddle
     from openvino.tools.mo import convert_model

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
     ov_model = convert_model(exe, example_input=[x], example_output=[y])


.. important::

   The ``convert_model()`` method returns ``ov.Model`` that you can optimize, compile, or save to a file for subsequent use.


Supported PaddlePaddle Layers
#############################

For the list of supported standard layers, refer to the :doc:`Supported Operations <../../../../../about-openvino/compatibility-and-support/supported-operations>` page.

Frequently Asked Questions (FAQ)
################################

The model conversion API displays explanatory messages for typographical errors, incorrectly used options, or other issues. They describe the potential cause of the problem and give a link to the :doc:`Model Optimizer FAQ <../[legacy]-model-optimizer-faq>`, which provides instructions on how to resolve most issues. The FAQ also includes links to relevant sections in :doc:`Convert a Model <../../legacy-conversion-api>` to help you understand what went wrong.

Additional Resources
####################

See the :doc:`Model Conversion Tutorials <[legacy]-conversion-tutorials>` page for a set of tutorials providing step-by-step instructions for converting specific PaddlePaddle models.


