Setting Input Shapes
====================


.. meta::
   :description: Learn how to increase the efficiency of a model by providing an additional
                 shape definition with the ``input`` parameter of ``openvino.convert_model``
                 and ``ovc``.


``openvino.convert_model`` supports conversion of models with dynamic input shapes that
contain undefined dimensions. However, if the shape of data is not going to change from
one inference request to another, it is recommended to **set up static shapes**
(all dimensions are fully defined) for the inputs, using the the ``input`` parameter.
Doing so at the model preparation stage, not at runtime, can be beneficial in terms of
performance and memory consumption.

For more information on changing input shapes in runtime, refer to the
:doc:`Changing input shapes <../running-inference/changing-input-shape>` guide.
To learn more about dynamic shapes in runtime, refer to the
:doc:`Dynamic Shapes <../running-inference/dynamic-shapes>` guide. To download models,
you can visit `Hugging Face <https://huggingface.co/models>`__.

The OpenVINO Runtime API may present certain limitations in inferring models with undefined
dimensions on some hardware. See the :doc:`Feature support matrix <../../about-openvino/compatibility-and-support/supported-devices>`
for reference. In this case, the ``input`` parameter and the
:doc:`reshape method <../running-inference/changing-input-shape>` can help to resolve undefined
dimensions.

For example, run model conversion for the TensorFlow MobileNet model with the single input
and specify the input shape of ``[2,300,300,3]``:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         import openvino as ov
         ov_model = ov.convert_model("MobileNet.pb", input=[2, 300, 300, 3])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc MobileNet.pb --input [2,300,300,3]

If a model has multiple inputs, the input shape should be specified in ``input`` parameter
as a list. In ``ovc``, this is a command separate list, and in ``openvino.convert_model``
this is a Python list or tuple with number of elements matching the number of inputs in
the model. Use input names from the original model to define the mapping between inputs
and shapes specified. The following example demonstrates the conversion of the ONNX OCR
model with a pair of inputs ``data`` and ``seq_len`` and specifies shapes ``[3,150,200,1]``
and ``[3]`` for them respectively:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         import openvino as ov
         ov_model = ov.convert_model("ocr.onnx", input=[("data", [3,150,200,1]), ("seq_len", [3])])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc ocr.onnx --input data[3,150,200,1],seq_len[3]

If the order of inputs is defined in the input model and the order is known for the user,
names could be omitted. In this case, it is important to specify shapes in the
same order of input model inputs:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         import openvino as ov
         ov_model = ov.convert_model("ocr.onnx", input=([3,150,200,1], [3]))

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc ocr.onnx --input [3,150,200,1],[3]

Whether the model has a specified order of inputs depends on the original framework.
Usually, it is convenient to set shapes without specifying the names of the parameters
in the case of PyTorch model conversion because a PyTorch model is considered as
a callable that usually accepts positional parameters. On the other hand, names of inputs
are convenient when converting models from model files, because naming of inputs is
a good practice for many frameworks that serialize models to files.

The ``input`` parameter allows overriding original input shapes if it is supported by
the model topology. Shapes with dynamic dimensions in the original model can be replaced
with static shapes for the converted model, and vice versa. The dynamic dimension can be
marked in model conversion API parameter as ``-1`` or ``?`` when using ``ovc``.
For example, launch model conversion for the ONNX OCR model and specify dynamic batch
dimension for inputs:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         import openvino as ov
         ov_model = ov.convert_model("ocr.onnx", input=[("data", [-1, 150, 200, 1]), ("seq_len", [-1])])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc ocr.onnx --input "data[?,150,200,1],seq_len[?]"

To optimize memory consumption for models with undefined dimensions in run-time,
model conversion API provides the capability to define boundaries of dimensions.
The boundaries of undefined dimension can be specified with ellipsis in the command
line or with ``openvino.Dimension`` class in Python.
For example, launch model conversion for the ONNX OCR model and specify a boundary for
the batch dimension 1..3, which means that the input tensor will have batch dimension
minimum 1 and maximum 3 in inference:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         import openvino as ov
         batch_dim = ov.Dimension(1, 3)
         ov_model = ov.convert_model("ocr.onnx", input=[("data", [batch_dim, 150, 200, 1]), ("seq_len", [batch_dim])])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc ocr.onnx --input data[1..3,150,200,1],seq_len[1..3]

In practice, not every model is designed in a way that allows change of input shapes.
An attempt to change the shape for such models may lead to an exception during model
conversion, later in model inference, or even to wrong results of inference without
explicit exception raised. A knowledge about model topology is required to set
shapes appropriately.

