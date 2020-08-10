# (Deprecated) Case Study: Converting SSD Models Created with TensorFlow* Object Detection API {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_TensorFlow_SSD_ObjectDetection_API}

This is a deprecated page. Please, consider reading [this](../convert_model/tf_specific/Convert_Object_Detection_API_Models.md) page describing new approach to convert Object Detection API models giving closer to TensorFlow inference results.

## Converting Models Created with TensorFlow Object Detection API Version prior 1.6.0

As explained in the [Sub-graph Replacement in Model Optimizer](Subgraph_Replacement_Model_Optimizer.md) section, there are multiple
ways to setup the sub-graph matching. In this example we are focusing on the defining the sub-graph via a set of
"start" and "end" nodes.
The result of matching is two buckets of nodes:
* Nodes "between" start and end nodes.
* Nodes connected to the first list, but just on the constant path (e.g. these nodes are not connected to the inputs of the entire graph).

Let's look closer to the SSD models from the TensorFlow* detection model
<a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">zoo</a>:
[SSD MobileNet](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz) and
[SSD InceptionV2](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz).

*   Nodes "between" start and end nodes
*   Nodes connected to the first list, but just on the constant path (for example, these nodes are not connected to the inputs of the entire graph). Let's look closer to the SSD models from the TensorFlow\* detection model <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">zoo</a> : [SSD MobileNet](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz) and [SSD InceptionV2](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz).

A distinct layer of any SSD topology is the `DetectionOutput` layer. This layer is implemented with a dozens of primitive operations in TensorFlow, while in Inference Engine, it is one [layer](../../../ops/opset.md). Thus, to convert a SSD model from the TensorFlow, the Model Optimizer should replace the entire sub-graph of operations that implement the `DetectionOutput` layer with a single well-known `DetectionOutput` node.

The Inference Engine `DetectionOutput` layer consumes three tensors in the following order:

1.  Tensor with locations of bounding boxes
2.  Tensor with confidences for each bounding box
3.  Tensor with prior boxes (anchors in TensorFlow terminology)

`DetectionOutput` layer produces one tensor with seven numbers for each actual detection. There are more output tensors in the TensorFlow Object Detection API, but the values in them are consistent with the Inference Engine ones.

The difference with [other examples](Subgraph_Replacement_Model_Optimizer.md) is that here the `DetectionOutput` sub-graph is replaced with a new sub-graph (not a single layer).

Look at sub-graph replacement configuration file `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/legacy_ssd_support.json` that is used to enable two models listed above:
```json
[  
    {  
        "custom_attributes": {  
            "code_type": "caffe.PriorBoxParameter.CENTER_SIZE",  
            "confidence_threshold": 0.01,  
            "keep_top_k": 200,  
            "nms_threshold": 0.45,  
            "pad_mode": "caffe.ResizeParameter.CONSTANT",  
            "resize_mode": "caffe.ResizeParameter.WARP"  
        },  
        "id": "TFObjectDetectionAPIDetectionOutput",  
        "include_inputs_to_sub_graph": true,  
        "include_outputs_to_sub_graph": true,  
        "instances": {  
            "end_points": [  
                "detection_boxes",  
                "detection_scores",  
                "num_detections"  
            ],  
            "start_points": [  
                "Postprocessor/Shape",  
                "Postprocessor/Slice",  
                "Postprocessor/ExpandDims",  
                "Postprocessor/Reshape_1"  
            ]  
        },  
        "match_kind": "points"  
    },
    {
        "custom_attributes": {
        },
        "id": "PreprocessorReplacement",
        "inputs": [
            [
                {
                    "node": "map/Shape$",
                    "port": 0
                },
                {
                    "node": "map/TensorArrayUnstack/Shape$",
                    "port": 0
                },
                {
                    "node": "map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3$",
                    "port": 2
                }
            ]
        ],
        "instances": [
            ".*Preprocessor/"
        ],
        "match_kind": "scope",
        "outputs": [
            {
                "node": "sub$",
                "port": 0
            },
            {
                "node": "map/TensorArrayStack_1/TensorArrayGatherV3$",
                "port": 0
            }
        ]
    }
]
```

**Key lines**:

*	Lines 3-10 define static attributes that will be saved to the Intermediate Representation `.xml` file for `DetectionOutput` layer.

*	Lines 12 and 13 define values for attributes that should be always set to "true" for this release of the Model Optimizer. These two attributes are specific for sub-graph match by points only.

*	Lines 14-26 define one instance of the sub-graph to be match. It is an important difference between sub-graph matching by scope and points. Several instances could be specified for matching by scope, but matching with points allows specifying just one instance. So the full node names (not regular expressions like in case of match with scope) are specified in `instances` dictionary.

The second sub-graph replacer with identifier `PreprocessorReplacement` is used to remove the `Preprocessing` block from the graph. The replacer removes all nodes from this scope except nodes performing mean value subtraction and scaling (if applicable). Implementation of the replacer is in the `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/Preprocessor.py` file.

Now let's analyze the structure of the topologies generated with the Object Detection API. There are several blocks in the graph performing particular task:

*   `Preprocessor` block resizes, scales and subtracts mean values from the input image.

*   `FeatureExtractor` block is a [MobileNet](https://arxiv.org/abs/1704.04861) or other backbone to extract features.

*   `MultipleGridAnchorGenerator` block creates initial bounding boxes locations (anchors).

*   `Postprocessor` block acts as a `DetectionOutput` layer. So we need to replace `Postprocessor` block with `DetectionOutput` layer. It is necessary to add all input nodes of the `Postprocessor` scope to the list `start_points`. Consider inputs of each of these nodes:

	*   `Postprocessor/Shape` consumes tensor with locations.
	*   `Postprocessor/Slice` consumes tensor with confidences.
	*   `Postprocessor/ExpandDims` consumes tensor with prior boxes.
	*   `Postprocessor/Reshape_1` consumes tensor with locations similarly to the `Postprocessor/Shape` node. Despite the fact that the last node `Postprocessor/Reshape_1` gets the same tensor as node `Postprocessor/Shape`, it must be explicitly put to the list.

Object Detection API `Postprocessor` block generates output nodes: `detection_boxes`, `detection_scores`, `num_detections`, `detection_classes`.

Now consider the implementation of the sub-graph replacer, available in the `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/SSDs.py`. The file is rather big, so only some code snippets are used:
```python
class PostprocessorReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'TFObjectDetectionAPIDetectionOutput'
```

These lines define the new `PostprocessorReplacement` class inherited from `FrontReplacementFromConfigFileSubGraph`. `FrontReplacementFromConfigFileSubGraph` is designed to replace sub-graph of operations described in the configuration file. There are methods to override for implementing custom replacement logic that we need:

*   `generate_sub_graph` performs new sub-graph generation and returns dictionary where key is an alias name for the node and value is a Node objects. The dictionary has the same format as parameter `match` in the `replace_sub_graph` method in the example with <a href="Subgraph_Replacement_Model_Optimizer.html#replace-using-isomorphism-pattern">networkx sub-graph isomorphism pattern</a>. This dictionary is passed as argument to the next three methods, so it should contain entries the for nodes that the functions need.

*   `input_edges_match` specifies mapping between input edges to sub-graph before replacement and after replacement. The key of the dictionary is a tuple specifying input tensor of the sub-graph before replacement: sub-graph input node name and input port number for this node. The value for this key is also a tuple specifying the node where this tensor should be attached during replacement: the node name (or alias name of the node) and the input port for this node. If the port number is zero, the parameter could be omitted so the key or value is just a node name (alias). Default implementation of the method returns an empty dictionary, so Model Optimizer does not create new edges.

*   `output_edges_match` returns mapping between old output edges of the matched nodes and new sub-graph node and output edge index. The format is similar to the dictionary returned in the `input_edges_match` method. The only difference is that instead of specifying input port numbers for the nodes it is necessary to specify output port number. Of course, this mapping is needed for the output nodes only. Default implementation of the method returns an empty dictionary, so the Model Optimizer does not create new edges.

*   `nodes_to_remove` specifies list of nodes that Model Optimizer should remove after sub-graph replacement. Default implementation of the method removes all sub-graph nodes.

Review of the replacer code, considering details of the `DetectionOutput` layer implementation in the Inference Engine. There are several constraints to the input tensors of the `DetectionOutput` layer:

*   The tensor with locations must be of shape `[#&zwj;batch, #&zwj;prior_boxes * 4]` or `[#&zwj;batch, #&zwj;prior_boxes * 5]` depending on shared locations between different batches or not.
*   The tensor with confidences must be of shape `[#&zwj;batch, #&zwj;prior_boxes * #&zwj;classes]` and confidences values are in range [0, 1], that is passed through `softmax` layer.
*   The tensor with prior boxes must be of shape `[#&zwj;batch, 2, #&zwj;prior_boxes * 4]`. Inference Engine expects that it contains variance values which TensorFlow Object Detection API does not add.

To enable these models, add `Reshape` operations for locations and confidences tensors and update the values for the prior boxes to include the variance constants (they are not there in TensorFlow Object Detection API).

Look at the `generate_sub_graph` method:
```python
def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
    log.debug('PostprocessorReplacement.generate_sub_graph')
    log.debug('matched_nodes = {}'.format(match.matched_nodes_names()))
    # softmax to be applied to the confidence
    softmax_conf_op = Softmax(graph, {'axis': 2, 'nchw_layout': True})
    softmax_conf_node = softmax_conf_op.add_node(dict(name='DetectionOutput_SoftMax_conf_'))
    # Inference Engine DetectionOutput layer consumes flattened tensors
    # reshape operation to flatten locations tensor
    reshape_loc_op = Reshape(graph, {'dim': np.array([0, -1])})
    reshape_loc_node = reshape_loc_op.add_node(dict(name='DetectionOutput_Reshape_loc_'))
    # Inference Engine DetectionOutput layer consumes flattened tensors
    # reshape operation to flatten confidence tensor
    reshape_conf_op = Reshape(graph, {'dim': np.array([0, -1])})
    reshape_conf_node = reshape_conf_op.add_node(dict(name='DetectionOutput_Reshape_conf_'))
    # create Node object from Op class
    detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
    detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
    detection_output_op.attrs['infer'] = __class__.do_infer
    detection_output_node = detection_output_op.add_node(dict(name=detection_output_op.attrs['type'] + '_'))
    # create internal edges of the sub-graph. In this case we add edges to connect input port 0 and 1 of the
    # detection output with output of reshape of locations and reshape of confidence
    create_edge(softmax_conf_node, reshape_conf_node, 0, 0)
    create_edge(reshape_loc_node, detection_output_node, 0, 0)
    create_edge(reshape_conf_node, detection_output_node, 0, 1)
    return {'detection_output_node': detection_output_node, 'reshape_conf_node': softmax_conf_node,
            'reshape_loc_node': reshape_loc_node}
```
The method has two inputs: the graph to operate on and the instance of `SubgraphMatch` object, which describes matched sub-graph. The latter class has several useful methods to get particular input/output node of the sub-graph by input/output index or by node name pattern. Examples of these methods usage are given below.

**Key lines**:

*	Lines 6 and 7 create new instance of operation of type `Softmax` and graph Node object corresponding to that operation.

*	Lines 11-12 and 16-17 create new instance of operation of type `Reshape` to reshape locations and confidences tensors correspondingly.

*	Lines 20-23 create new instance of operation `DetectionOutput` and graph Node object corresponding to that operation.

*	Lines 27-29 connect `softmax` node with `reshape` node and connect two reshaped locations and confidences tensors with `DetectionOutput` node.

*	Lines 30-31 define dictionary with aliases for detection output node, reshape locations and confidences nodes. These aliases are used in the `input_edges_match` and `output_edges_match` methods.

The `input_edges_match` method is the following:
```python
def input_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
    locs_consumer_node, locs_consumer_node_port = match.input_nodes(0)[0]
    conf_consumer_node, conf_consumer_node_port = match.input_nodes(1)[0]
    priors_consumer_node, priors_consumer_node_port = match.input_nodes(2)[0]
    # create matching nodes for locations and confidence tensors using simple scheme "old_node_name: new_node_name"
    # which in fact means "(old_node_name, 0): (new_node_name, 0)", while first '0' means old_port and the second
    # zero defines 'new_port'.
    return {locs_consumer_node.id: new_sub_graph['reshape_loc_node'].id,
            conf_consumer_node.id: new_sub_graph['reshape_conf_node'].id,
            priors_consumer_node.id: (new_sub_graph['detection_output_node'].id, 2),
            }
```
The method has three parameters: input `graph`, `match` object describing matched sub-graph and `new_sub_graph` dictionary with alias names returned from the `generate_sub_graph` method.

**Key lines**:

*	Lines 2-4 initialize Node objects and input ports for the 	nodes where the input tensors for the sub-graph are consumed. The method `match.input_nodes(ind)` returns list of tuples where the first element is a Node object and the second is the input port for this node which consumes the ind-th input tensor of the sub-graph. `input_points` list in the configuration file defines the order of input tensors to the sub-graph. For example, the `locs_consumer_node` object of type Node is a node that consumes tensor with locations in the port with number `locs_consumer_node_port`.

*	Lines 8-11 define dictionary with the mapping of tensors as described above. Note that the attribute `id` of the Node object contains the name of the node in the graph.

The `output_edges_match` method is the following:
```python
def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
    # the DetectionOutput in IE produces single tensor, but in TF it produces two tensors, so we need to create only
    # one output edge match
    return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}
```

The method has the same three parameters as `input_edges_match` method. The returned dictionary contains mapping just for one tensor initially produces by the first output node of the sub-graph (which is `detection_boxes` according to the configuration file) to a single output tensor of the created `DetectionOutput` node. In fact, it is possible to use any output node of the initial sub-graph in mapping, because the sub-graph output nodes are the output nodes of the whole graph (their output is not consumed by any other nodes).

Now, the Model Optimizer knows how to replace the sub-graph. The last step to enable the model is to cut-off some parts of the graph not needed during inference.

It is necessary to remove the `Preprocessor` block where image is resized. Inference Engine does not support dynamic input shapes, so the Model Optimizer must froze the input image size, and thus, resizing of the image is not necessary. This is achieved by replacer `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/Preprocessor.py` which is executed automatically.

There are several `Switch` operations in the `Postprocessor` block without output edges. For example:
```sh
Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/switch_t
```
```sh
Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/switch_f
```
```sh
Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_1/cond/switch_t
```
```sh
Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_1/cond/switch_f
```

Model Optimizer marks these nodes as output nodes of the topology. Some parts of the `Posprocessor` blocks are not removed during sub-graph replacement because of that. In order to fix this issue, it is necessary to specify output nodes of the graph manually using the `--output` command line parameter.

###Example Model Optimizer Command-Line for TensorFlow\* SSD

The final command line to convert SSDs from the TensorFlow Object Detection API Zoo is:
```shell
./mo_tf.py --input_model=<path_to_frozen.pb> --tensorflow_use_custom_operations_config extensions/front/tf/legacy_ssd_support.json --output="detection_boxes,detection_scores,num_detections"
```

## Converting MobileNet V2 model created with TensorFlow Object Detection API <a name="convert_mobilenet_v2"></a>
The [MobileNet V2 model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) differs from the previous version, so converting the model requires a new sub-graph replacement configuration file and new command line parameters. The major differences are:

* The `Preprocessor` block has two outputs: the pre-processed image and the pre-processed image size.
* The `Postprocessor` block has one more input (in comparison with models created with TensorFlow Object Detection API
version 1.6 or lower): the pre-processed image size.
* Some node names have been changed in the `Postprocessor` block.

The updated sub-graph replacement configuration file `extensions/front/tf/ssd_v2_support.json` reflecting these changes
is the following:

```json
[
    {
        "custom_attributes": {
            "code_type": "caffe.PriorBoxParameter.CENTER_SIZE",
            "confidence_threshold": 0.01,
            "keep_top_k": 200,
            "nms_threshold": 0.6,
            "pad_mode": "caffe.ResizeParameter.CONSTANT",
            "resize_mode": "caffe.ResizeParameter.WARP"
        },
        "id": "TFObjectDetectionAPIDetectionOutput",
        "include_inputs_to_sub_graph": true,
        "include_outputs_to_sub_graph": true,
        "instances": {
            "end_points": [
                "detection_boxes",
                "detection_scores",
                "num_detections"
            ],
            "start_points": [
                "Postprocessor/Shape",
                "Postprocessor/scale_logits",
                "Postprocessor/ExpandDims",
                "Postprocessor/Reshape_1",
                "Postprocessor/ToFloat"
            ]
        },
        "match_kind": "points"
    },
    {
        "custom_attributes": {
        },
        "id": "PreprocessorReplacement",
        "inputs": [
            [
                {
                    "node": "map/Shape$",
                    "port": 0
                },
                {
                    "node": "map/TensorArrayUnstack/Shape$",
                    "port": 0
                },
                {
                    "node": "map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3$",
                    "port": 2
                }
            ]
        ],
        "instances": [
            ".*Preprocessor/"
        ],
        "match_kind": "scope",
        "outputs": [
            {
                "node": "sub$",
                "port": 0
            },
            {
                "node": "map/TensorArrayStack_1/TensorArrayGatherV3$",
                "port": 0
            }
        ]
    }
]
```

### Example of Model Optimizer Command-Line for TensorFlow SSD MobileNet V2
The final command line to convert MobileNet SSD V2 from the TensorFlow Object Detection Zoo is the following:

```sh
./mo_tf.py --input_model=<path_to_frozen.pb> --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json --output="detection_boxes,detection_scores,num_detections"
```
