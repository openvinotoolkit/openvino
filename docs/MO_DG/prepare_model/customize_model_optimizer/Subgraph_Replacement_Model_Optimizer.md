# Sub-Graph Replacement in the Model Optimizer  {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Subgraph_Replacement_Model_Optimizer}

The document has been deprecated. Refer to the [Model Optimizer Customization](Subgraph_Replacement_Model_Optimizer.md) for the up-to-date documentation.


### Replace Sub-graph of Operations Using Nodes Name Pattern

TensorFlow uses a mechanism of scope to group related operation nodes. It is a good practice to put nodes performing particular task into the scope. This approach divides a graph into logical blocks that are easier to review in TensorBoard\*. The `scope`, in fact, just defines a common prefix for the node names in the scope.

For example, Inception topologies contain several types of so-called "Inception blocks". Some of them are exactly equal to each other, but located in different places of the network. For example, Inception V4 from `tensorflow.contrib.slim` module has inception blocks `Mixed_5b`, `Mixed_5c` and `Mixed_5d` with exactly the same nodes with the same attributes.

Now consider situation when someone implemented these Inception blocks extremely efficiently using single Inference Engine custom layer called `InceptionBlock` and would like to replace these blocks with instances of the layer to decrease inference time. Model Optimizer provides mechanism to replace sub-graph of operations defined by the regular expressions for the node names prefixes (scope). In this particular case, some of the patterns are: `.*InceptionV4/Mixed_5b`, `.*InceptionV4/Mixed_5c` and `.*InceptionV4/Mixed_5d`. Each pattern starts with `.*`, because a prefix `InceptionV4` is added to all nodes names during a model freeze.

The sub-graph replacement using nodes name pattern is a bit trickier than replacements of single operation and networkx isomorphism pattern described above. You should do the following additional steps in comparison with previously described replacements:

1.  Prepare configuration file template defining node names patterns and information about custom layer attributes.

2.  Run Model Optimizer with command line parameter to add information about input and output nodes of the specified sub-graphs.

Consider the following possible configuration file for the Inception Block replacer:
```json
[
    {
        "custom_attributes": {
            "attr1_key": "attr1_value",
            "attr2_key": 123456
        },
        "id": "InceptionBlockReplacer",
        "op": "InceptionBlock",
        "instances": [
            ".*InceptionV4/Mixed_5b",
            ".*InceptionV4/Mixed_5c",
            ".*InceptionV4/Mixed_5d"
        ],
        "match_kind": "scope"
    }
]
```
The `.json` file contains list of dictionaries. Each dictionary defines one replacement. Each replacement is defined with several keys:

*   `id` (mandatory) is a unique identifier of the replacer. It is used in the Python\* code that implements sub-graph replacement to link the class and the replacement description from the configuration file.

*   `match_kind` (mandatory) is a string that specifies what matching algorithm is used. Currently supported `scope` and `points`. In this example, the first one is considered. The `points` match kind is described below.

*   `instances` (mandatory) specifies instances of the sub-graph to be matched. It contains a list of node names prefixes patterns for the match kind `scope`.

*   `custom_attributes` (optional) is a dictionary with static attributes of the layer to be dumped to Inference Engine Intermediate Representation `.xml` file.

*   `op` (optional) is used only if the sub-graph replacement Python code is not needed, because the sub-graph should be replaced with a single node of type `op`. If this attribute is not set, it is necessary to implement Python code with sub-graph generation code. Both options are considered in this example.

When the configuration file is ready, run the Model Optimizer with regular command line parameters pointing to the file with model and input shapes (if necessary) and additional parameter `--tensorflow_custom_operations_config_update` pointing to the generated configuration file. If the file is correct, Model Optimizer adds two keys to the `InceptionBlockReplacer` dictionary: `inputs` and `outputs` with the following content:
```json
[
    {
        "id": "InceptionBlockReplacer",
        ...
        "inputs": [
            [
                {
                    "node": "Branch_2/Conv2d_0a_1x1/Conv2D$",
                    "port": 0
                },
                {
                    "node": "Branch_3/AvgPool_0a_3x3/AvgPool$",
                    "port": 0
                },
                {
                    "node": "Branch_1/Conv2d_0a_1x1/Conv2D$",
                    "port": 0
                },
                {
                    "node": "Branch_0/Conv2d_0a_1x1/Conv2D$",
                    "port": 0
                }
            ]
        ],
        "outputs": [
            {
                "node": "concat$",
                "port": 0
            }
        ]
    }
]
```
The value for key `inputs` is a list of lists describing input tensors of the sub-graph. Each element of the top-level list corresponds to one unique input tensor of the sub-graph. Each internal list describes a list of nodes consuming this tensor and port numbers where the tensor is consumed. Model Optimizer generates regular expressions for the input nodes names to uniquely identify them in each instance of the sub-graph defined by the `instances`. Denote these nodes as input nodes of the sub-graph.

In the InceptionV4 topology, the `InceptionV4/Mixed_5b` block has four input tensors from outside of the sub-graph, but all of them are produced by the node `InceptionV4/Mixed_5a/concat`. Therefore, the top-level list of the `inputs` contains one list corresponding to this tensor. Four input nodes of the sub-graph consume the tensor produced by `InceptionV4/Mixed_5a/concat` node. In this case, all four input nodes consume input tensor into port 0.

The order of items in the internal list describing nodes does not matter, but the order of elements in the top-level list is important. This order defines the order in which the Model Optimizer attaches input tensors to a new generated node if the sub-graph is replaced with a single node. The i-th input node of the sub-graph is obtained using call `match.single_input_node(i)` in the sub-graph replacer code. More information about API is given below. If you need to change the order of input tensors, you can edit the configuration file in the text-editor.

The value for the key `outputs` is a list describing nodes of the sub-graph producing tensor that goes outside of the sub-graph or does not have child nodes. Denote these nodes as output nodes of the sub-graph. The order of elements in the list is important. The i-th element of the list describes the i-th output tensor of the sub-graph, which could be obtained using call `match.output_node(i)`. The order of elements can be manually changed in the configuration file. Model Optimizer uses this order to connect output edges if the sub-graph is replaced with a single node.

Now, when meaning of `inputs` and `outputs` attributes is clean, return back to the replacer implementation. The replacer `InceptionBlockReplacer` contains attribute `op` with the value `InceptionBlock`, which means that the identified sub-graph should be replaced with a single layer of type `InceptionBlock`. This layer is not known for the Model Optimizer, so it is necessary to define it. See [Extending the Model Optimizer with New Primitives](Extending_Model_Optimizer_with_New_Primitives.md). You must create file `extension/ops/InceptionBlock.py` with the following content:
```py
import numpy as np
from mo.graph.graph import Node
from mo.ops.op import Op
class InceptionBlock(Op):
    op = "InceptionBlock"
    enabled = True
    def __init__(self, graph, attrs):
        super().__init__(graph, attrs, {
            'type': __class__.op,
            'op': __class__.op,
        })
```
The shape inference function is not defined. In this case, Model Optimizer uses TensorFlow fallback to calculate shapes of the sub-graph output tensors.

Run the Model Optimizer with the regular command line parameters, path to the model file and input shape (if necessary), and the parameter `--tensorflow_use_custom_operations_config` and point to the created configuration file. Model Optimizer generates Intermediate Representation `.xml` file with three sequential layers of type `InceptionBlock` like in the following example:
```xml
<layer id="1658" name="InceptionBlock1877" precision="FP32" type="InceptionBlock">
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>384</dim>
            <dim>35</dim>
            <dim>35</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>384</dim>
            <dim>35</dim>
            <dim>35</dim>
        </port>
    </output>
</layer>
```
The implementation of the sub-graph replacement by scope with a single layer is complete. The next subsection explains 
how Model Optimizer replaces sub-graph identified by start/end nodes (`points`) with another sub-graph.

### <a name="sub_graph_replacement_using_points"></a> Replace Sub-graph of Operations Using Points
In this scenario, for the matching algorithm user defines the sub-graph via a set of "start" and "end" nodes.
Given the set, the Model Optimizer performs the following steps:
1. Starts graph traversal from every _start_ nodes following the direction of the graph edges.
The search stops in _end_   nodes or in case of nodes without further children. All visited nodes are added to the matched sub-graph.
2. Starts another graph traversal from each non-start node of the sub-graph, i.e. every node except nodes from "start" set.
In this step the edges are traversed in the opposite edge direction. All newly visited nodes are added to the
   matched sub-graph. This step is needed to add nodes required for calculation values of internal nodes of the
   matched sub-graph.
3. Checks that all "end" nodes were reached from "input" nodes. If no then exit with error.
4. Check that there are no "Placeholder" operations among added nodes. If it is not true then some side branch of
   the sub-graph (added in step 2) depends on inputs of the network. Such configuration is not correct so exit with error.

This algorithm finds all nodes "between" start and end nodes. Also nodes needed for calculation of non-input nodes of the
matched sub-graph produce _constant_ values because they do not depend on input of the network.
**This sub-graph match has a limitation that each start node must have only one input**. Therefore, it is not possible
to specify, for example, convolution node as input because it has two inputs: data tensor and tensor with weights.

For example of replacement with points, please refer to the case-study of the 
[Converting TensorFlow* Object Detection API Models](../convert_model/tf_specific/Convert_Object_Detection_API_Models.md)
