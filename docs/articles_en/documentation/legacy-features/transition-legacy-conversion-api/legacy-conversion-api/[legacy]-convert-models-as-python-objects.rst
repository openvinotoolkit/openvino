[LEGACY] Convert Models Represented as Python Objects
=============================================================

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Model Preparation <../../../../openvino-workflow/model-preparation>` article.

Model conversion API is represented by ``convert_model()`` method in openvino.tools.mo namespace. ``convert_model()`` is compatible with types from openvino.runtime, like PartialShape, Layout, Type, etc.

``convert_model()`` has the ability available from the command-line tool, plus the ability to pass Python model objects, such as a PyTorch model or TensorFlow Keras model directly, without saving them into files and without leaving the training environment (Jupyter Notebook or training scripts). In addition to input models consumed directly from Python, ``convert_model`` can take OpenVINO extension objects constructed directly in Python for easier conversion of operations that are not supported in OpenVINO.

.. note::

   Model conversion can be performed only when you install
   :doc:`the development tools <../../../legacy-features/install-dev-tools>`, which provide
   both the ``convert_model()`` method and ``mo`` command-line tool.
   The functionality from this article is applicable for ``convert_model()`` only and it is
   not present in command-line tool.


``convert_model()`` returns an openvino.runtime.Model object which can be compiled and inferred or serialized to IR.

Example of converting a PyTorch model directly from memory:

.. code-block:: py
   :force:

   import torchvision
   from openvino.tools.mo import convert_model

   model = torchvision.models.resnet50(weights='DEFAULT')
   ov_model = convert_model(model)

The following types are supported as an input model for ``convert_model()``:

* PyTorch - ``torch.nn.Module``, ``torch.jit.ScriptModule``, ``torch.jit.ScriptFunction``. Refer to the :doc:`Converting a PyTorch Model <[legacy]-supported-model-formats/[legacy]-convert-pytorch>` article for more details.
* TensorFlow / TensorFlow 2 / Keras - ``tf.keras.Model``, ``tf.keras.layers.Layer``, ``tf.compat.v1.Graph``, ``tf.compat.v1.GraphDef``, ``tf.Module``, ``tf.function``, ``tf.compat.v1.session``, ``tf.train.checkpoint``. Refer to the :doc:`Converting a TensorFlow Model <[legacy]-supported-model-formats/[legacy]-convert-tensorflow>` article for more details.

``convert_model()`` accepts all parameters available in the MO command-line tool. Parameters can be specified by Python classes or string analogs, similar to the command-line tool.

Example of using native Python classes to set ``input_shape``, ``mean_values`` and ``layout``:

.. code-block:: py
   :force:

   from openvino.runtime import PartialShape, Layout
   from openvino.tools.mo import convert_model

   ov_model = convert_model(model, input_shape=PartialShape([1,3,100,100]), mean_values=[127, 127, 127], layout=Layout("NCHW"))

Example of using strings for setting ``input_shape``, ``mean_values`` and ``layout``:

.. code-block:: py
   :force:

   from openvino.runtime import Layout
   from openvino.tools.mo import convert_model

   ov_model = convert_model(model, input_shape="[1,3,100,100]", mean_values="[127,127,127]", layout="NCHW")


The ``input`` parameter can be set by a ``tuple`` with a name, shape, and type. The input name of the type string is required in the tuple. The shape and type are optional.
The shape can be a ``list`` or ``tuple`` of dimensions (``int`` or ``openvino.runtime.Dimension``), or ``openvino.runtime.PartialShape``, or ``openvino.runtime.Shape``. The type can be of numpy type or ``openvino.runtime.Type``.

Example of using a tuple in the ``input`` parameter to cut a model:

.. code-block:: py
   :force:

   from openvino.tools.mo import convert_model

   ov_model = convert_model(model, input=("input_name", [3], np.float32))

For complex cases, when a value needs to be set in the ``input`` parameter, the ``InputCutInfo`` class can be used. ``InputCutInfo`` accepts four parameters: ``name``, ``shape``, ``type``, and ``value``.

``InputCutInfo("input_name", [3], np.float32, [0.5, 2.1, 3.4])`` is equivalent of ``InputCutInfo(name="input_name", shape=[3], type=np.float32, value=[0.5, 2.1, 3.4])``.

Supported types for ``InputCutInfo``:

* name: ``string``.
* shape: ``list`` or ``tuple`` of dimensions (``int`` or ``openvino.runtime.Dimension``), ``openvino.runtime.PartialShape``, ``openvino.runtime.Shape``.
* type: ``numpy type``, ``openvino.runtime.Type``.
* value: ``numpy.ndarray``, ``list`` of numeric values, ``bool``.

Example of using ``InputCutInfo`` to freeze an input with value:

.. code-block:: py
   :force:

   from openvino.tools.mo import convert_model, InputCutInfo

   ov_model = convert_model(model, input=InputCutInfo("input_name", [3], np.float32, [0.5, 2.1, 3.4]))

To set parameters for models with multiple inputs, use ``list`` of parameters.
Parameters supporting ``list``:

* input
* input_shape
* layout
* source_layout
* dest_layout
* mean_values
* scale_values

Example of using lists to set shapes, types and layout for multiple inputs:

.. code-block:: py
   :force:

   from openvino.runtime import Layout
   from openvino.tools.mo import convert_model, LayoutMap

   ov_model = convert_model(model, input=[("input1", [1,3,100,100], np.float32), ("input2", [1,3,100,100], np.float32)], layout=[Layout("NCHW"), LayoutMap("NCHW", "NHWC")])

``layout``, ``source_layout`` and ``dest_layout`` accept an ``openvino.runtime.Layout`` object or ``string``.

Example of using the ``Layout`` class to set the layout of a model input:

.. code-block:: py
   :force:

   from openvino.runtime import Layout
   from openvino.tools.mo import convert_model

   ov_model = convert_model(model, source_layout=Layout("NCHW"))

To set both source and destination layouts in the ``layout`` parameter, use the ``LayoutMap`` class. ``LayoutMap`` accepts two parameters: ``source_layout`` and ``target_layout``.

``LayoutMap("NCHW", "NHWC")`` is equivalent to ``LayoutMap(source_layout="NCHW", target_layout="NHWC")``.

Example of using the ``LayoutMap`` class to change the layout of a model input:

.. code-block:: py
   :force:

   from openvino.tools.mo import convert_model, LayoutMap

   ov_model = convert_model(model, layout=LayoutMap("NCHW", "NHWC"))

Example of using the ``serialize`` method to save the converted model to OpenVINO IR:

.. code-block:: py
   :force:

   from openvino.runtime import serialize

   serialize(ov_model, "model.xml")

