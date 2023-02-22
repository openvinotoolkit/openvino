# Model Optimizer Operation {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions_Model_Optimizer_Operation}

Model Optimizer defines a `mo.ops.Op` class (`Op` will be used later in the document to be short), which is a base class
for an operation used in the Model Optimizer. The instance of the `Op` class serves several purposes:

1. Stores the operation attributes.
2. Stores the operation shape/value and type inference functions.
3. Defines operation attributes to be saved to the corresponding IR section.
4. Contains convenient methods to create a graph node from an `Op` object instance and connect it with the existing
graph.
5. Used in the extractors to store parsed attributes and operation specific attributes in the dedicated graph node.

It is important to mention that there is no connection between the instance of the `Op` class and the `Node` object
created from it. The `Op` class is just a container for attributes describing the operation. Model Optimizer uses the `Op`
class during a model conversion to create a node of the graph with attributes copied from the `Op` class instance. Graph
manipulations are performed with graph `Node`s and their attributes and does not involve `Op`s.

There are a number of common attributes used in the operations. Below is the list of these attributes with description.

* `id` — **(Mandatory)** — unique identifier of a node in a graph. Generated automatically, equal to the number of nodes in the graph plus
1 if not specified.
* `name` — **(Mandatory)** — name of the operation. Generated automatically, equal to the `id` if not specified.
* `type` — **(Mandatory)** —  type of the operation according to the [opset specification](@ref openvino_docs_ops_opset). For the internal
Model Optimizer operations, this attribute should be set to `None`. The model conversion fails if an operation with
`type` equal to `None` comes to the IR emitting phase.
* `version` — **(Mandatory)** —  the operation set (opset) name the operation belongs to. If not specified, 
Model Optimizer sets it equal to `experimental`. For more information about operation sets, refer to 
[OpenVINO Model Representation](@ref openvino_docs_OV_UG_Model_Representation) section. 
* `op` — Model Optimizer type of the operation. In many cases, the value of `type` is equal to the value of `op`. However,
when Model Optimizer cannot instantiate the opset operation during model loading, it creates an instance of an internal
operation. Thus, the attribute `op` is used as a type of this internal operation. Later in the pipeline, the node created
from an internal operation will be replaced during front, middle or back phase with node(s) created from the opset.
* `infer` — the attribute defines a function calculating output tensor(s) shape and optional value(s). The attribute
may be set to `None` for the internal Model Optimizer operations used during the front phase only. For more information 
about the shape inference function, refer to the [Partial Inference](@ref mo-partial-inference).
* `type_infer` — the attribute defines a function calculating output tensor(s) data type. If the attribute is not
defined, the default function is used. The function checks if the `data_type` node attribute is set and then
propagates this type to the output tensor from the "port 0". Otherwise, it propagates the data type of the tensor coming
into the input "port 0" to the output tensor from the "port 0".
* `in_ports_count` — default number of input ports to be created for the operation. Additional ports can be created or
redundant ports can be removed using dedicated `Node` class API methods.
* `out_ports_count` — default number of output ports to be created for the operation. Additional ports can be created or
redundant ports can be removed using dedicated `Node` class API methods.

Below is an example of the Model Optimizer class for the [SoftMax](../../../ops/activation/SoftMax_1.md) operation from
the `mo/ops/softmax.py` file with the comments in code.

```py
class Softmax(Op):
    # The class attribute defines a name of the operation so the operation class can be obtained using the
    # "Op.get_op_class_by_name()" static method
    op = 'SoftMax'

    # The operation works as an extractor by default. This is a legacy behavior, currently not recommended for use,
    # thus "enabled" class attribute is set to False. The recommended approach is to use dedicated extractor extension.
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {  # The constructor of the base class Op is called with additional default attributes.
            'type': __class__.op,  # The operation is from the opset so the type is set to 'SoftMax'.
            'op': __class__.op,  # Internal Model Optimizer operation has the same type.
            'version': 'opset1',  # The operation corresponds to opset1.
            'infer': Softmax.infer,  # Shape inference function is defined below.
            'axis': 1,  # Default value for the "axis" attribute of the operation SoftMax.
            'in_ports_count': 1,  # The operation has one input.
            'out_ports_count': 1,  # The operation produces one output.
        }, attrs)

    # The method returns operation specific attributes list. This method is important when implementing
    # extractor inherited from CaffePythonFrontExtractorOp class to extract attribute for Caffe Python operation.
    # However, it is currently used interchangeably with the "backend_attrs()" method. If the "backend_attrs()" is not used,
    # then the "supported_attrs()" is used instead. In this particular case, the operation has just one attribute "axis".
    def supported_attrs(self):
        return ['axis']

    @staticmethod
    def infer(node: Node):
        "some code calculating output shape and values"
```

There is a dedicated method called `backend_attrs()` defining a list of attributes to be saved to the IR. Consider an
example from the `mo/ops/pooling.py` file:
```py
   def backend_attrs(self):
        return [
            ('strides', lambda node: ','.join(map(str, node['stride'][node.spatial_dims]))),
            ('kernel', lambda node: ','.join(map(str, node['window'][node.spatial_dims]))),

            ('pads_begin', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 0)))),
            ('pads_end', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 1)))),

            ('pool-method', 'pool_method'),
            ('exclude-pad', 'exclude_pad'),

            'rounding_type',
            'auto_pad',
        ]
```

The `backend_attrs()` function returns a list of records. A record can be of one of the following formats:
1. A string defining the attribute to be saved to the IR. If the value of the attribute is `None`, the attribute is
not saved. Examples of this case are `rounding_type` and `auto_pad`.
2. A tuple, where the first element is a string defining the name of the attribute as it will appear in the IR and the
second element is a function to produce the value for this attribute. The function gets an instance of the `Node` as the
only parameter and returns a string with the value to be saved to the IR. Examples of this case are `strides`, `kernel`,
`pads_begin` and `pads_end`.
3. A tuple, where the first element is a string defining the name of the attribute as it will appear in the IR and the
second element is the name of the `Node` attribute to get the value from. Examples of this case are `pool-method` and
`exclude-pad`.

## Additional Resources

* [Model Optimizer Extensibility](Customize_Model_Optimizer.md)
* [Graph Traversal and Modification Using Ports and Connections](Model_Optimizer_Ports_Connections.md)
* [Model Optimizer Extensions](Model_Optimizer_Extensions.md)
* [Extending Model Optimizer with Caffe Python Layers](Extending_Model_Optimizer_with_Caffe_Python_Layers.md)
