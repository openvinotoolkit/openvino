# Extending the Model Optimizer with New Primitives {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Extending_Model_Optimizer_with_New_Primitives}

This section explains how to register a custom layer in the Model Optimizer, including how to register Proposal as a custom layer. This section also demonstrates how `Proposal` works as a custom layer.

Model Optimizer loads the model, goes through the topology, and tries to find each layer type in the list of known layers. If the Model Optimizer does not find a layer in that list, it looks for the layer in the list of custom layers. If the Model Optimizer fails to find the layer among the defined custom layers, it registers a Caffe\* fallback for for the output shape inference. If the Model Optimizer does not find Caffe and cannot infer shapes, the Model Optimizer fails with an appropriate message.

You must know two things about custom layers with the Model Optimizer:

*   How to map a subgraph in a FW model to a subgraph consisting of Inference Engine layers. For Caffe, the subgraph is a 1-to-1 mapping of a Caffe layer to an Inference Engine layer.
*   How to infer shapes for unknown subgraphs. This can be either for a step in which the internal representation consists of framework-specific layers, or for a step in which the internal representation consists of Inference Engine layers.

You also have the option of a framework fallback for unknown subgraphs, for when the original framework is used for inference of output shapes of operations. The example below demonstrates the case in which the framework is not available or should not be used.

## Preparing an Example Topology

> **NOTE**: Skip this section if you have a topology with a layer that is not known to the Model Optimizer.

The information in this section prepares a Caffe\* model with the provided, deployment-ready `prototxt` for a
well-known topology called
[Faster-R-CNN protoxt](https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt)
to demonstrate the workflow. To use this example, you must have
[weights and biases](http://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0) for inference,
because `prototxt` just describes the structure of the topology.

1.  Download the `.caffemodel` and `.prototxt` files
2.  Run the Model Optimizer on the `.caffemodel` and `.prototxt` files:
```shell
python mo.py --input_model VGG16_faster_rcnn_final.caffemodel --input_proto test.prototxt
```
You will likely see the error message:
```shell
Error parsing text-format caffe.NetParameter: 196:16: Message type "caffe.DropoutParameter" has no field named "scale_train".
```
Whether you see the error depends on your Caffe version. For example, BVLC Caffe does not support the boolean parameter `scale_train` for the `dropout` layer. The error message does not matter, because the dropout layer is needed only for training, and the Model Optimizer removes it.
3.  To proceed, comment out these lines in `test.prototxt`:
```sh
...
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "fc6"
  top: "fc6"
  dropout_param {
    dropout_ratio: 0.5
    # scale_train: false # <-------------- comment out this line
  }
}
...
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "fc7"
  top: "fc7"
  dropout_param {
    dropout_ratio: 0.5
    # scale_train: false # <-------------- comment out this line
  }
}
...
```
4.  Run the Model Optimizer on this model again:
```shell
python mo.py --input_model VGG16_faster_rcnn_final.caffemodel --input_proto test.prototxt
```
    You get the model successfully converted to Intermediate Representation, and you can infer it with the Inference Engine.

    However, the aim of this tutorial is to demonstrate the way of supporting custom layers not yet supported by the Model Optimizer.
    If you want to understand better how Model Optimizer works, remove the extension for layer `Proposal` and follow all steps of this tutorial.

5.	Remove the extension for layer `Proposal`:
```sh
mkdir extensions/old
mv extensions/front/caffe/proposal_python_ext.py extensions/old/proposal_python_ext_old.py
mv extensions/ops/proposal_python_example.py extensions/old/proposal_python__example_old.py
```
6.	Now you can run the Model Optimizer on this model once again:
```sh
python mo.py --input_model VGG16_faster_rcnn_final.caffemodel --input_proto test.prototxt
```
You will see the message:
```shell
[ ERROR ]  Found custom layer proposal. Model Optimizer does not support this layer.
Please, register it in CustomLayersMapping.xml or implement extension.
For more information please refer to Model Optimizer FAQ, question #FAQ45.
```
This message means the Model Optimizer can load the model, but is unable to infer the shape and handle the custom layer properties.

## Registering a Custom Layer as a Model Optimizer Extension

In the following sections, you will learn how to make the Model Optimizer independent from Caffe\* when processing a
model that has a custom layer. In this example, the custom layer is referred to as the Proposal layer.

Use this section to implement the mapping rules for the `Proposal` layer attributes and the output shape calculation. As part of these steps, you must first create a class for the `Proposal` layer and inherit it from general-purpose Op that defines the interface of every new custom layer.

In this section, it is important to understand the `Op` class and its function. The implementation of this class shows that it expects a graph and attributes to be passed when initializing. The graph and attributes are in `<INSTALL_DIR>/deployment_tools/model_optimizer/mo/ops/op.py`

`Op` keeps the attributes for each operation and contains logic for handling node creation for internal model representation. `Op` is responsible for dumping each particular operation to the `.xml` format for the Intermediate Representation. By inheriting from it, the technical items are complete and you concentrate on the specificity of this layer: the attributes it supports and the rules on computing its output shape.

Follow these steps:

1.  Create the file `python_proposal.py` in the directory `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/ops`:
```python
from mo.ops.op import Op
class PythonProposalOp(Op):
    pass
```
2.  Define the name of the operation and make a stub constructor:
```python
from mo.ops.op import Op
class PythonProposalOp(Op):
    op = 'Proposal'
    def __init__(self, graph, attrs):
        super().__init__(graph)
```
3.  Every `Op` must have three specific fields defined: `type`, `op`, and `infer`. In most cases, the `type` and `op` names are the same, and `infer` is defined as a function to compute the output shape. Reflect these fields in your constructor:
```python
from mo.ops.op import Op
class PythonProposalOp(Op):
    op = 'Proposal'
    def __init__(self, graph, attrs):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': None
        }
        super().__init__(graph, mandatory_props, attrs)
```
 According to the Intermediate Representation catalog, Proposal layer has the following attributes:

    *   `pre_nms_topn`
    *   `post_nms_topn`
    *   `nms_thresh`
    *   `feat_stride`
    *   `min_size`
    *   `base_size`
    *   `ratio`
    *   `scale`
4.  In defining supported attribute names, it is best to use the same names as in the original models. The names are similar to parameters and have no connection with the model layer properties. For clarity, you can use the name `my_ratio` for `ratio`. Other than defining the list of supported parameters, you can define only the parameters that appear in the Intermediate Representation in the `backend_attrs` method.  
    Define your attributes:
```python
class PythonProposalOp(Op):
    # ... constructor
     def supported_attrs(self):
            return [
                'pre_nms_topn',
                'post_nms_topn',
                'nms_thresh',
                'feat_stride',
                'min_size',
                'base_size',
                'ratio',
                'scale'
            ]
```
5.  Model Optimizer now knows how to create the layer called Proposal when it is in the topology and what attributes this layer has. However, the Model Optimizer does not know how to calculate the output shape of this operation. Define a rule to calculate the output shape:
```python
import numpy as np
from mo.graph.graph import Node
from mo.ops.op import Op
class PythonProposalOp(Op):
   def __init__(self, graph, attrs):
       mandatory_props = {
           'type': __class__.op,
           'op': __class__.op,
           'infer': PythonProposalOp.calculate_output_shape
       }
       super().__init__(graph, mandatory_props, attrs)
    # ... supported attrs
    @staticmethod
    def calculate_output_shape(node: Node):
        node.out_node().shape = (1, 1, 1, 1) # any Proposal now has always the same output
```
6.  According to the Intermediate Representation catalog, Proposal layer has the following output calculation formula, where shape dynamically depends on the `post_nms_topn` parameter.  
    Implement the output calculation formula in Python\*:
```python
import numpy as np
class PythonProposalOp(Op):
    # ... static fields
    # ... constructor
    # ... supported attrs
    @staticmethod
    def calculate_output_shape(node: Node):
        input_shape = node.in_node(0).shape
        out_shape = np.array([0, 0], dtype=np.int64)
        # rois blob: holds R regions of interest, each is a 5 - tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle(x1, y1, x2, y2)
        out_shape[0] = input_shape[0] * node.post_nms_topn
        out_shape[1] = 5
        node.out_node(0).shape = out_shape
```
    The node does not contain this parameter because it should be initialized in the constructor and in other parameters. The Inference Engine contains the implementation of a Caffe\*-like Proposal layer and works well with the default values from `caffe.proto`:
```
// Message that stores parameters used by ProposalLayer message ProposalParameter { optional uint32 feat_stride = 1 [default = 16]; optional uint32 base_size = 2 [default = 16]; optional uint32 min_size = 3 [default = 16]; repeated float ratio = 4; repeated float scale = 5; optional uint32 pre_nms_topn = 6 [default = 6000]; optional uint32 post_nms_topn = 7 [default = 300]; optional float nms_thresh = 8 [default = 0.7]; }
```
7.  Change the constructor as follows:
```python
class PythonProposalOp(Op):
    # ... static fields
    def __init__(self, graph, attrs):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'feat_stride': 16,
            'base_size': 16,
            'min_size': 16,
            'ratio': [0.5, 1, 2],
            'scale': [8, 16, 32],
            'pre_nms_topn': 6000,
            'post_nms_topn': 300,
            'nms_thresh': 0.7,
            'infer': PythonProposalOp.calculate_output_shape
        }
        super().__init__(graph, mandatory_props, attrs)
    # ... supported attrs
    # ... calculate output shape

```

It is mandatory to call two functions right after the implementation of that class:

```
class ProposalPythonOp(Op):
      ...

register_caffe_python_extractor(ProposalPythonOp, 'rpn.proposal_layer.ProposalLayer')
Op.excluded_classes.append(ProposalPythonOp)
```

Note that the first call <code>register_caffe_python_extractor(ProposalPythonOp, 'rpn.proposal_layer.ProposalLayer')</code> registers the extension of the layer in the Model Optimizer that will be found by a specific name (it is mandatory to join module name and layer name): <code>'rpn.proposal_layer.ProposalLayer'</code>.

The second call prevents the Model Optimizer from using this extension as if it is an extension for a layer with type `Proposal`. Otherwise, this layer can be chosen as an implementation of extension that can lead to potential issues.

**Summary**

In this section you implemented support for a custom layer with type `Python` that is `Proposal` layer in the topology. You learned how to calculate output shape of this layer.

The values of attributes are hardcoded, and in the next section you will learn how to extract these values from original framework model (Caffe model in this case).

## Registering Rules to Pass Extension Layer Properties from a Caffe\* Model to the Intermediate Representation

Model Optimizer now knows how to set the shape of the `PythonProposalOp` operation, but it is incorrect to initialize attributes with same values for every operation. Instead, the values should be extracted from the original topology. Model Optimizer does not know how to map the custom layer properties to the `PythonProposalOp`. For this, you must register the `FrontExtractorOp` instance.

> **NOTE**: This step is required only if the layer requires parameters from the original model.

1.	Remove call functions `register_caffe_python_extractor` and `Op.excluded_classes.append` from the file with `op`, because you will implement extracted attributes from prototxt by yourself.
There are multiple types of layers in Caffe: for example, `Convolution` and `Pooling`. Also, there is a specific type for custom Python\* layers called `Python`. Therefore, it is necessary to distinguish between those 'usual' types of layers and custom ones. If you want to implement extensions for a layer with type different to `Python`, you need to inherit your class of operation (for example, `ProposalFrontExtractor`) from `FrontExtractorOp`. Otherwise, inherit your class of operation from `CaffePythonFrontExtractorOp`.
2.  Create a file `python_proposal_ext.py` in the folder `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/caffe`
```py
from mo.front.extractor import CaffePythonFrontExtractorOp
class PythonProposalFrontExtractor(CaffePythonFrontExtractorOp):
    pass
```
For other layers types, inherit from `FrontExtractorOp`:
```py
	from mo.front.extractor import FrontExtractorOp
	class ProposalFrontExtractor(FrontExtractorOp):
		pass
```
You will implement extractor for layer with type `Python`, however, the steps are generally the same for layers with other types.
3.  Specify the operation that the extractor refers to and a specific flag. The flag represents whether the operation should be used by the Model Optimizer or should be excluded from processing:
```py
from mo.front.extractor import CaffePythonFrontExtractorOp
class PythonProposalFrontExtractor(CaffePythonFrontExtractorOp):
    op = 'rpn.proposal_layer.ProposalLayer'
    enabled = True
```
4.  Register a mapping rule between the original model and the `PythonProposalOp` attributes by overriding the following function:
```py
from mo.front.extractor import CaffePythonFrontExtractorOp
from mo.ops.op import Op
class ProposalPythonFrontExtractor(CaffePythonFrontExtractorOp):
    op = 'rpn.proposal_layer.ProposalLayer'
    enabled = True
    @staticmethod
    def extract(node):
        proto_layer = node.pb
        param = proto_layer.python_param # each layer has a specific parameter, take a look at caffe.proto
        python_params = str(param.param_str) # for Python layers, all params are in param_str
        attrs = {
            'feat_stride': int(python_params.split(':')[-1])
        }
        # update the attributes of the node
        Op.get_op_class_by_name('Proposal').update_node_stat(node, attrs) # <------ here goes the name ('Proposal') of the Operation that was implemented before
        return __class__.enabled
```
> **NOTE:** if you implement extension for layer with type different to `Python`, change the following line: <code>Op.get_op_class_by_name('Proposal').update_node_stat(node, attrs)</code> to this line: <code>Op.get_op_class_by_name(__class__.op).update_node_stat(node, mapping_rule)</code>.
You have successfully extracted the parameter `feat_stride` from `prototxt`, assuming it is the only parameter in this layer.
5.  To increase the implementation flexibility:
```py
  from mo.front.extractor import CaffePythonFrontExtractorOp
  from mo.ops.op import Op
  class PythonProposalFrontExtractor(CaffePythonFrontExtractorOp):
      op = 'rpn.proposal_layer.ProposalLayer'
      enabled = True
      @staticmethod
      def extract(node):
          param = node.pb.python_param
          attrs = CaffePythonFrontExtractorOp.parse_param_str(param.param_str)
          Op.get_op_class_by_name('Proposal').update_node_stat(node, attrs)
          return ProposalPythonFrontExtractor.enabled
```

You can successfully convert the model. Open the `.xml` file and view your code:
```xml
...
<layer id="42" name="proposal" precision="FP32" type="Python">
    <data base_size="16" feat_stride="16" min_size="16" nms_thresh="0.7" post_nms_topn="300" pre_nms_topn="6000" ratio="[0.5, 1, 2]" scale="[8, 16, 32]"/>
   <input>
        <port id="0">
            <dim>1</dim>
            <dim>18</dim>
            <dim>15</dim>
            <dim>15</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>36</dim>
            <dim>15</dim>
            <dim>15</dim>
        </port>
        <port id="2">
            <dim>1</dim>
            <dim>3</dim>
        </port>
     </input>
     <output>
        <port id="3">
            <dim>300</dim>
            <dim>5</dim>
        </port>
    </output>
</layer>
...
```

Look at the output shape of the custom layer you implemented. The shape was calculated according to the rules specified in `PythonProposalOp`. The `ratio` and `scale` properties have the value `[0.5, 1, 2]` and `[8, 16, 32]`. They have square brackets because they are originally a repeated parameter. You converted the parameter to a list in `PythonProposalOp`. Model Optimizer cast the value to a string. According to Python\* rules, a list has a string representation of opening and closing square brackets and values joined by commas.

This is not a valid notation for the Intermediate Representation specification, because repeated parameters must be separated by a comma but without the brackets. Therefore, you must override the Model Optimizer default behavior regarding how it handles those parameters during the Intermediate Representation emitting stage, after the optimizations are complete. To do so, implement `backend_attrs()` in the `PythonProposalOp` class:
```python
class PythonProposalOp(Op):
    ... other methods
    def backend_attrs(self) -> list:
        """
        Gets list of attributes that should appear in resulting IR
        Returns:
            list of attributes names or list of tuples (name of attribute, pre-processing rule)
        """
        return [
            (  # a tuple per attribute
                'ratio',  # name of attribute
                # pre-processing rule in a form of lambda
                # lambda takes a PythonProposalOp node with all defined properties
                # it translates [1,2,3] -> "1,2,3"
                lambda node: ','.join(map(str, node['ratio']))
            ),
            (
                'scale',
                lambda node: ','.join(map(str, node['scale']))
            ),
            'feat_stride',
            'base_size',
            'min_size',
            'pre_nms_topn',
            'post_nms_topn',
            'nms_thresh'
            ]
```
The model can now be successfully converted.

Open the `.xml` file. `ratio` and `scale` have the expected correct values `0.5,1,2` and `8,16,32`:
```xml
    ...

    <layer id="33" name="proposal" precision="FP32" type="Python">
        <data base_size="16" feat_stride="16" min_size="16" nms_thresh="0.7" post_nms_topn="300" pre_nms_topn="6000" ratio="0.5,1,2" scale="8,16,32"/>
        <input>
          ...
        </input>
        <output>
           ...
        </output>
    </layer>

    ...
```

> **NOTE**: Model Optimizer supports the Faster-R-CNN topology. Run the following command for the same Intermediate Representation:

```sh
python mo.py --input_model VGG16_faster_rcnn_final.caffemodel --input_proto test.prototxt --extensions <INSTALL_DIR>/deployment_tools/inference-engine/samples/object_detection_sample/fasterrcnn_extensions
```

**Summary**

In this section you learned how to:

1.  Create a framework-independent extension implementation of the Intermediate Representation custom layer with unified logic for calculating output shapes, specified set of attributes
2.  Use the Framework-Specific property extractor to map original model custom layer properties to the expected properties of the Framework-Independent extension
3.  Manipulate the custom layer properties representation in the resulting Intermediate Representation

Files used in this section:

*   `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/ops/python_proposal.py`:

```py
import networkx as nx
import numpy as np
from mo.front.extractor import attr_getter
from mo.graph.graph import Node
from mo.ops.op import Op

class ProposalOp(Op):
    op = 'Proposal'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'post_nms_topn': 300,  # default in caffe-shared
            'infer': ProposalOp.proposal_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'feat_stride',
            'base_size',
            'min_size',
            'ratio',
            'scale',
            'pre_nms_topn',
            'post_nms_topn',
            'nms_thresh'
        ]

    def backend_attrs(self):
        return [
            'feat_stride',
            'base_size',
            'min_size',
            ('ratio', lambda node: attr_getter(node, 'ratio')),
            ('scale', lambda node: attr_getter(node, 'scale')),
            'pre_nms_topn',
            'post_nms_topn',
            'nms_thresh',
        ]

    @staticmethod
    def proposal_infer(node: Node):
        input_shape = node.in_node(0).shape
        out_shape = np.array([0, 0], dtype=np.int64)
        # rois blob: holds R regions of interest, each is a 5 - tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle(x1, y1, x2, y2)
        out_shape[0] = input_shape[0] * node.post_nms_topn
        out_shape[1] = 5
        node.out_node(0).shape = out_shape
```
*   `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/caffe/python_proposal_ext.py`:

```py
from mo.front.extractor import CaffePythonFrontExtractorOp
from mo.ops.op import Op

class ProposalPythonFrontExtractor(CaffePythonFrontExtractorOp):
    op = 'rpn.proposal_layer.ProposalLayer'
    enabled = True

    @staticmethod
    def extract(node):
        param = node.pb.python_param
        attrs = CaffePythonFrontExtractorOp.parse_param_str(param.param_str)
        Op.get_op_class_by_name('Proposal').update_node_stat(node, attrs)
        return ProposalPythonFrontExtractor.enabled
```
