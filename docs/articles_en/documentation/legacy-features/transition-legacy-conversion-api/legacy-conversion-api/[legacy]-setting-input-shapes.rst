[LEGACY] Setting Input Shapes
====================================

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the  :doc:`Setting Input Shapes <../../../../openvino-workflow/model-preparation/setting-input-shapes>` article.

With model conversion API you can increase your model's efficiency by providing an additional shape definition, with these two parameters: `input_shape` and `static_shape`.


.. meta::
   :description: Learn how to increase the efficiency of a model with MO by providing an additional shape definition with the input_shape and static_shape parameters.


Specifying input_shape parameter
################################

``convert_model()`` supports conversion of models with dynamic input shapes that contain undefined dimensions.
However, if the shape of data is not going to change from one inference request to another,
it is recommended to set up static shapes (when all dimensions are fully defined) for the inputs.
Doing it at this stage, instead of during inference in runtime, can be beneficial in terms of performance and memory consumption.
To set up static shapes, model conversion API provides the ``input_shape`` parameter.
For more information on input shapes under runtime, refer to the :doc:`Changing input shapes <../../../../openvino-workflow/running-inference/changing-input-shape>` guide.
To learn more about dynamic shapes in runtime, refer to the :doc:`Dynamic Shapes <../../../../openvino-workflow/running-inference/dynamic-shapes>` guide.

The OpenVINO Runtime API may present certain limitations in inferring models with undefined dimensions on some hardware.
In this case, the ``input_shape`` parameter and the :doc:`reshape method <../../../../openvino-workflow/running-inference/changing-input-shape>` can help to resolve undefined dimensions.

For example, run model conversion for the TensorFlow MobileNet model with the single input
and specify the input shape of ``[2,300,300,3]``:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         from openvino.tools.mo import convert_model
         ov_model = convert_model("MobileNet.pb", input_shape=[2,300,300,3])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         mo --input_model MobileNet.pb --input_shape [2,300,300,3]


If a model has multiple inputs, ``input_shape`` must be used in conjunction with ``input`` parameter.
The ``input`` parameter contains a list of input names, for which shapes in the same order are defined via ``input_shape``.
For example, launch model conversion for the ONNX OCR model with a pair of inputs ``data`` and ``seq_len``
and specify shapes ``[3,150,200,1]`` and ``[3]`` for them:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         from openvino.tools.mo import convert_model
         ov_model = convert_model("ocr.onnx", input=["data","seq_len"], input_shape=[[3,150,200,1],[3]])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         mo --input_model ocr.onnx --input data,seq_len --input_shape [3,150,200,1],[3]


Alternatively, specify input shapes, using the ``input`` parameter as follows:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         from openvino.tools.mo import convert_model
         ov_model = convert_model("ocr.onnx", input=[("data",[3,150,200,1]),("seq_len",[3])])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         mo --input_model ocr.onnx --input data[3,150,200,1],seq_len[3]


The ``input_shape`` parameter allows overriding original input shapes to ones compatible with a given model.
Dynamic shapes, i.e. with dynamic dimensions, can be replaced in the original model with static shapes for the converted model, and vice versa.
The dynamic dimension can be marked in model conversion API parameter as ``-1`` or ``?``.
For example, launch model conversion for the ONNX OCR model and specify dynamic batch dimension for inputs:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         from openvino.tools.mo import convert_model
         ov_model = convert_model("ocr.onnx", input=["data","seq_len"], input_shape=[[-1,150,200,1],[-1]]

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         mo --input_model ocr.onnx --input data,seq_len --input_shape [-1,150,200,1],[-1]


To optimize memory consumption for models with undefined dimensions in run-time, model conversion API provides the capability to define boundaries of dimensions.
The boundaries of undefined dimension can be specified with ellipsis.
For example, launch model conversion for the ONNX OCR model and specify a boundary for the batch dimension:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         from openvino.tools.mo import convert_model
         from openvino.runtime import Dimension
         ov_model = convert_model("ocr.onnx", input=["data","seq_len"], input_shape=[[Dimension(1,3),150,200,1],[Dimension(1,3)]]

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         mo --input_model ocr.onnx --input data,seq_len --input_shape [1..3,150,200,1],[1..3]


Practically, some models are not ready for input shapes change.
In this case, a new input shape cannot be set via model conversion API.
For more information about shape follow the :doc:`inference troubleshooting <[legacy]-troubleshooting-reshape-errors>`
and :ref:`ways to relax shape inference flow <how-to-fix-non-reshape-able-model>` guides.

Additional Resources
####################

* :doc:`Convert a Model <../legacy-conversion-api>`
* :doc:`Cutting Off Parts of a Model <[legacy]-cutting-parts-of-a-model>`

