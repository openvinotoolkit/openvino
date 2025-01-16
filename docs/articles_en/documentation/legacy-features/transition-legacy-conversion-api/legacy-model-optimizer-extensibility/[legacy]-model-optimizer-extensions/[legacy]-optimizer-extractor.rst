[LEGACY] Operation Extractor
=============================

.. meta::
   :description: Learn about a deprecated generic extension in Model Optimizer,
                 which provides the operation extractor usable for all model
                 frameworks.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated TensorFlow conversion method. The guide on the new and recommended method, using a new frontend, can be found in the  :doc:`Frontend Extensions <../../../../openvino-extensibility/frontend-extensions>` article.

Model Optimizer runs specific extractor for each operation in the model during the model loading.

There are several types of Model Optimizer extractor extensions:

1. The generic one, which is described in this article.
2. The special extractor for Caffe models with Python layers. This kind of extractor is described in the :doc:`Extending Model Optimizer with Caffe Python Layers <../[legacy]-extending-model-optimizer-with-caffe-python-layers>` guide.

Generic extension provides a generic mechanism for the operation extractor applicable for all frameworks. Model Optimizer provides the ``mo.front.extractor.FrontExtractorOp`` class as a base class to implement the extractor. It has the ``extract`` class method, which gets the only parameter ``Node``, which corresponds to the graph node to extract data from. The operation description in the original framework format is stored in the attribute ``pb`` of the node. The extractor goal is to parse this attribute and save necessary attributes to the corresponding node of the graph. Consider the extractor for the ``Const`` TensorFlow operation (refer to the ``extensions/front/tf/const_ext.py`` file):

.. code-block:: py
   :force:

   from openvino.tools.mo.front.extractor import FrontExtractorOp
   from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor, tf_tensor_shape, tf_tensor_content
   from openvino.tools.mo.ops.const import Const


   class ConstExtractor(FrontExtractorOp):
       # The "op" class attribute defines a type of the operation in the framework (in this case it is a TensorFlow),
       # for which the extractor should be triggered.
       op = 'Const'
       enabled = True  # The flag that indicates that this extractor is enabled.

       @classmethod
       def extract(cls, node):  # The entry point of the extractor.
           # The `node.pb` attribute stores the TensorFlow representation of the operation, which is a Protobuf message of the
           # specific format. In particular, the message contains the attribute called "value" containing the description of
           # the constant. The string "pb.attr["value"].tensor" is just a Python binding for Protobuf message parsing.
           pb_tensor = node.pb.attr["value"].tensor
           # Get the shape of the tensor from the protobuf message, using the helper function "tf_tensor_shape".
           shape = tf_tensor_shape(pb_tensor.tensor_shape)
           # Create a dictionary with necessary attributes.
           attrs = {
               'shape': shape,
               # Get the tensor value, using "tf_tensor_content" helper function.
               'value': tf_tensor_content(pb_tensor.dtype, shape, pb_tensor),
               # Get the tensor data type, using "tf_dtype_extractor" helper function.
               'data_type': tf_dtype_extractor(pb_tensor.dtype),
           }
           # Update the node attributes, using default attributes from the "Const" operation and attributes saved to the
           # "attrs" dictionary.
           Const.update_node_stat(node, attrs)
           return cls.enabled

Consider another example with an extractor of the ``Constant`` ONNX operation (refer to the ``extensions/front/onnx/const_ext.py`` file):

.. code-block:: py
   :force:

   from onnx import numpy_helper
   from onnx.numpy_helper import to_array

   from openvino.tools.mo.front.extractor import FrontExtractorOp
   from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr
   from openvino.tools.mo.ops.const import Const


   class ConstantExtractor(FrontExtractorOp):
       op = 'Constant'
       enabled = True

       @classmethod
       def extract(cls, node):
           # Use "onnx_attr" helper method, which parses the Protobuf representation of the operation saved in the "node".
           # Gets the value of the attribute with name "value" as "TensorProto" type (specified with a keyword "t").
           pb_value = onnx_attr(node, 'value', 't')
           # Use "numpy_helper.to_array()" ONNX helper method to convert "TensorProto" object to a numpy array.
           value = numpy_helper.to_array(pb_value)

           attrs = {
               'data_type': value.dtype,
               'value': value,
           }
           # Update the node attributes, using default attributes from the "Const" operation and attributes saved to the
           # "attrs" dictionary.
           Const.update_node_stat(node, attrs)
           return cls.enabled

The extractors for operations from different frameworks work similarly. The only difference is in the helper methods used to parse operation attributes encoded with a framework-specific representation.

A common practice is to use ``update_node_stat()`` method of the dedicated ``Op`` class to update the node attributes. This method does the following:

1. Sets values for common attributes like ``op``, ``type``, ``infer``, ``in_ports_count``, ``out_ports_count``, ``version`` to values specific to the dedicated operation (``Const`` operation in this case).
2. Uses ``supported_attrs()`` and ``backend_attrs()`` methods, defined in the ``Op`` class to update specific node attribute ``IE``. The IR emitter uses the value stored in the ``IE`` attribute to pre-process attribute values and save them to IR.
3. Optionally sets additional attributes provided to the ``update_node_stat()`` function as a second parameter. Usually these attributes are parsed from the particular instance of the operation.

.. note::
   Model Optimizer uses numpy arrays to store values and numpy arrays of ``np.int64`` type to store shapes in the graph.

====================
Additional Resources
====================

* :doc:`Model Optimizer Extensibility <../../legacy-model-optimizer-extensibility>`
* :doc:`Graph Traversal and Modification Using Ports and Connections <../../legacy-model-optimizer-extensibility/[legacy]-graph-traversal-and-modification>`
* :doc:`Model Optimizer Extensions <../[legacy]-model-optimizer-extensions>`
* :doc:`Extending Model Optimizer with Caffe Python Layers <../[legacy]-extending-model-optimizer-with-caffe-python-layers>`

