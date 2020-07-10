#  Converting Faster R-CNN models, created with TensorFlow Object Detection API  {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_TensorFlow_Faster_RCNN_ObjectDetection_API}

This is a deprecated page. Please, consider reading [this](../convert_model/tf_specific/Convert_Object_Detection_API_Models.md) page describing new approach to convert Object Detection API models giving closer to TensorFlow inference results.

## Converting models created with TensorFlow Object Detection API version equal or higher than 1.6.0
This chapter describes how to convert selected Faster R-CNN models from the TensorFlow Object Detection API zoo version equal or higher than 1.6.0. The full list of supported models is provided in the table below. Note that currently batch size 1 is supported only. The only Inference Engine plugin supporting these topologies inference is CPU.

The Faster R-CNN models contain several building blocks similar to building blocks from SSD models so it is highly recommended to read chapter about [enabling TensorFlow Object Detection API SSD models](TensorFlow_SSD_ObjectDetection_API.md) first. Detailed information about Faster R-CNN topologies is provided [here](https://arxiv.org/abs/1506.01497).

The TensorFlow network consists of a number of big blocks grouped by scope:

*   `Preprocessor` performs scaling/resizing of the image and converts input data to [0, 1] interval. Has two outputs: the first one is modified input image and the second one is a constant tensor with shape (batch_size, 3) and values (resized_image_height, resized_image_width, 3).

*    `FirstStageFeatureExtractor` is a backbone feature extractor.

*    `FirstStageBoxPredictor` calculates boxes and classes predictions.

*    `GridAnchorGenerator`  generates anchors coordinates.

*    `ClipToWindow` crops anchors to the resized image size.

*    `Decode` decodes coordinates of boxes using anchors and data from the `FirstStageBoxPredictor`.

*    `BatchMultiClassNonMaxSuppression` performs non maximum suppression.

*    `map` scales coordinates of boxes to [0, 1] interval by dividing coordinates by (resized_image_height, resized_image_width).

*    `map_1` scales coordinates from [0, 1] interval to resized image sizes.

*    `SecondStageFeatureExtractor` is a feature extractor for predicted Regions of interest (ROIs).

*    `SecondStageBoxPredictor` refines box coordinates according `SecondStageFeatureExtractor`.

*    `SecondStagePostprocessor` is Detection Output layer performing final boxes predictions.

### Sub-graph replacements
There are three sub-graph replacements defined in the `extensions/front/tf/legacy_faster_rcnn_support.json` used to convert these models:

*   the first one replaces the `Preprocessor` block. The implementation of this replacer is in the `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/Preprocessor.py`

*   the second one replaces a number of blocks in the the graph including `GridAnchorGenerator`, `ClipToWindow`, `Decode`, `BatchMultiClassNonMaxSuppression`, `Tile`, `Tile_1` and `map` with Proposal and ROIRooling layers and some additional layers to pre-process input data

*   the third one replaces `SecondStagePostprocessor` with a DetectionOutput layer.

The second replacer is defined using the following configuration that matches sub-graph by points:

```json
    {
        "custom_attributes": {
            "nms_threshold": 0.7,
            "feat_stride": 16,
            "max_proposals": 100,
            "anchor_base_size": 256,
            "anchor_scales": [0.25, 0.5, 1.0, 2.0],
            "anchor_aspect_ratios": [0.5, 1.0, 2.0],
            "roi_spatial_scale": 0.0625
        },
        "id": "TFObjectDetectionAPIFasterRCNNProposalAndROIPooling",
        "include_inputs_to_sub_graph": true,
        "include_outputs_to_sub_graph": true,
        "instances": {
            "end_points": [
                "CropAndResize",
                "map_1/TensorArrayStack/TensorArrayGatherV3",
                "map_1/while/strided_slice/Enter",
                "BatchMultiClassNonMaxSuppression/map/TensorArrayStack_4/TensorArrayGatherV3"
            ],
            "start_points": [
                "FirstStageBoxPredictor/concat",
                "FirstStageBoxPredictor/concat_1",
                "GridAnchorGenerator/Identity",
                "Shape",
                "CropAndResize"
            ]
        },
        "match_kind": "points"
    }
```

The `start_points` list contains the following nodes:

*   `FirstStageBoxPredictor/concat` node produces box coordinates predictions.

*   `FirstStageBoxPredictor/concat_1` node produces classes predictions which will be used for the ROIs

*   `GridAnchorGenerator/Identity` node produces anchors coordinates.

*   `Shape` and `CropAndResize` nodes are specified as inputs to correctly isolate the required sub-graph. Refer to the [chapter](Subgraph_Replacement_Model_Optimizer.md) for more information about replacements by points.

The `end_points` list contains the following nodes:

*   `CropAndResize` is the node that performs ROI pooling operation.

*   `map_1/TensorArrayStack/TensorArrayGatherV3`, `map_1/while/strided_slice/Enter` and `BatchMultiClassNonMaxSuppression/map/TensorArrayStack_4/TensorArrayGatherV3` are specified to correctly isolate the sub-graph.

The `custom_attributes` dictionary contains attributes where most values are taken from the topology-specific configuration file `samples/configs/faster_rcnn_*.config` of the [TensorFlow Object Detection API repository](https://github.com/tensorflow/models/tree/master/research/object_detection):

*   `nms_threshold` is the value of the `first_stage_nms_iou_threshold` parameter.

*   `feat_stride` is the value of the `height_stride` and `width_stride` parameters. Inference Engine supports case when these two values are equal that is why the replacement configuration file contains just one parameter.

*   `max_proposals` is the value of the `max_total_detections` parameter which is a maximum number of proposal boxes from the Proposal layer and detected boxes.

*   `anchor_base_size` is the base size of the generated anchor. The 256 is the default value for this parameter and it is not specified in the configuration file.

*   `anchor_scales" is the value of the `scales` attrbite.

*   `anchor_aspect_ratios` is the value of the `aspect_ratios` attribute.

*   `roi_spatial_scale` is needed for the Inference Engine ROIPooling layer. It is the default value that is not actually used.

The identifier for this replacer is `TFObjectDetectionAPIFasterRCNNProposalAndROIPooling`. The Python implementation of this replacer is in the file `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/FasterRCNNs.py`.

The first four functions of the replacer class are the following:

```python
class TFObjectDetectionAPIFasterRCNNProposalAndROIPooling(FrontReplacementFromConfigFileSubGraph):
    """
    This class replaces sub-graph of operations with Proposal and ROIPooling layers and additional layers transforming
    tensors from layout of TensorFlow to layout required by Inference Engine.
    Refer to comments inside the function for more information about performed actions.
    """
    replacement_id = 'TFObjectDetectionAPIFasterRCNNProposalAndROIPooling'

    def run_after(self):
        return [PreprocessorReplacement]

    def run_before(self):
        return [SecondStagePostprocessorReplacement]

    def output_edges_match(self, graph: nx.DiGraph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['roi_pooling_node'].id}

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        new_list = match.matched_nodes_names().copy()
        # do not remove nodes that produce box predictions and class predictions
        new_list.remove(match.single_input_node(0)[0].id)
        new_list.remove(match.single_input_node(1)[0].id)
        return new_list
```

The function `run_after` returns list of Python classes inherited from one of the replacer classes (`FrontReplacementOp`, `FrontReplacementPattern`, `FrontReplacementFromConfigFileSubGraph` etc) those current sub-graph replacement class must be run after. In this case the replacer must be run after the `Preprocessor` is removed by the `PreprocessorReplacement` replacer. Similar way the `run_before` function is used to tell Model Optimizer to execute `SecondStagePostprocessorReplacement` before this replacer.

The `output_edges_match` function describes matching between the output nodes of the sub-graph before replacement and after. In this case the only needed output node of the sub-graph is the `CropAndResize` node which is identified with `match.output_node(0)[0]`. The new output node which is created in the `generate_sub_graph` function is identified with `new_sub_graph['roi_pooling_node']`.

The `nodes_to_remove` function takes the default list of nodes to be removed which contains all matched nodes and remove from them two input nodes which are identified with `match.single_input_node(0)[0]` and `match.single_input_node(1)[0]`. These nodes will be connected as inputs to new nodes being generated in the `generate_sub_graph` function so they should node be removed.

The code generating new sub-graph is the following:

```python
    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        log.debug('TFObjectDetectionAPIFasterRCNNProposal: matched_nodes = {}'.format(match.matched_nodes_names()))

        config_attrs = match.custom_replacement_desc.custom_attributes
        nms_threshold = config_attrs['nms_threshold']
        feat_stride = config_attrs['feat_stride']
        max_proposals = config_attrs['max_proposals']
        anchor_base_size = config_attrs['anchor_base_size']
        roi_spatial_scale = config_attrs['roi_spatial_scale']
        proposal_ratios = config_attrs['anchor_aspect_ratios']
        proposal_scales = config_attrs['anchor_scales']
        anchors_count = len(proposal_ratios) * len(proposal_scales)
```

These lines get parameters defined in the sub-graph replacement configuration file and calculate initial anchors count.

```python
        # get the ROIPool size from the CropAndResize which performs the same action
        if 'CropAndResize' not in graph.nodes():
            raise Error('Failed to find node with name "CropAndResize" in the topology. Probably this is not Faster'
                        ' RCNN topology or it is not supported')
        roi_pool_size = Node(graph, 'CropAndResize').in_node(3).value[0]
```

The code above gets the ROI Pooling spatial output dimension size as a value from the fourth argument of the node with name `CropAndResize`.

```python
        # Convolution/matmul node that produces classes predictions
        # Permute result of the tensor with classes permissions so it will be in a correct layout for Softmax
        predictions_node = match.single_input_node(1)[0].in_node(0).in_node(0)
        permute_predictions_op = Permute(graph, {'order': np.array([0, 2, 3, 1])})
        permute_predictions_node = permute_predictions_op.create_node([], dict(name=predictions_node.name + '/Permute_'))
        insert_node_after(predictions_node, permute_predictions_node, 0)

        reshape_classes_op = Reshape(graph, {'dim': np.array([0, -1, 2])})
        reshape_classes_node = reshape_classes_op.create_node([permute_predictions_node],
                                                              dict(name='Reshape_FirstStageBoxPredictor_Class_'))
        update_attrs(reshape_classes_node, 'shape_attrs', 'dim')

        softmax_conf_op = Softmax(graph, {'axis': 1})
        softmax_conf_node = softmax_conf_op.create_node([reshape_classes_node],
                                                        dict(name='FirstStageBoxPredictor_SoftMax_Class_'))
```

The output with class predictions from the `FirstStageBoxPredictor` is generated with a convolution operation. The convolution output data layout in TensorFlow is NHWC while Inference Engine uses NCHW layout. Model Optimizer by default converts the weights of TensorFlow convolutions to produce output tensor in NCHW layout required by Inference Engine. The issue arises because the class predictions tensor is passed through the Softmax operation to produce class probabilities. The Inference Engine Softmax is performed over the fastest-changing dimension which is 'W' in Inference Engine. Thus, the softmax operation will be performed over a wrong dimension after conversion of the convolution node producing classes predicitions. The solution is to add Permute and Reshape operations to prepare the input data for Softmax. The Reshape operation is required to make the size of the fastest-changing dimension equal to 2, because there are 2 classes being predicted: background and foreground.

Another issue is that layout of elements in the predicted classes tensor is different between TensorFlow and Inference Engine Proposal layer requirements. In TensorFlow the tensor has the following virtual layout [N, H, W, num_anchors, num_classes] while the Inference Engine Proposal layer requires in the following virtual layout [N, num_classes, num_anchors, H, W]. Thus, it is necessary to reshape, permute and then reshape again output from the Softmax to the required shape for the Proposal layer:

```python
        reshape_softmax_op = Reshape(graph, {'dim': np.array([1, anchors_count, 2, -1])})
        reshape_softmax_node = reshape_softmax_op.create_node([softmax_conf_node], dict(name='Reshape_Softmax_Class_'))
        update_attrs(reshape_softmax_node, 'shape_attrs', 'dim')

        permute_reshape_softmax_op = Permute(graph, {'order': np.array([0, 1, 3, 2])})
        permute_reshape_softmax_node = permute_reshape_softmax_op.create_node([reshape_softmax_node],
                                                                              dict(name='Permute_'))

        # implement custom reshape infer function because we need to know the input convolution node output dimension
        # sizes but we can know it only after partial infer
        reshape_permute_op = Reshape(graph, {'dim': np.ones([4]), 'anchors_count': anchors_count,
                                             'conv_node': predictions_node})
        reshape_permute_op.attrs['old_infer'] = reshape_permute_op.attrs['infer']
        reshape_permute_op.attrs['infer'] = __class__.classes_probabilities_reshape_shape_infer
        reshape_permute_node = reshape_permute_op.create_node([permute_reshape_softmax_node],
                                                              dict(name='Reshape_Permute_Class_'))
        update_attrs(reshape_permute_node, 'shape_attrs', 'dim')
```

The Proposal layer has 3 inputs: classes probabilities, boxes predictions and a input shape of the image. The first two tensors are ready so it is necessary to create the Const operation that produces the desired third input tensor.

```python
        # create constant input with the image height, width and scale H and scale W (if present) required for Proposal
        const_value = np.array([[input_height, input_width, 1]], dtype=np.float32)
        const_op = Const(graph, dict(value=const_value, shape=const_value.shape))
        const_node = const_op.create_node([], dict(name='Proposal_const_image_size_'))
```

Now add the Proposal layer:

```python

        proposal_op = ProposalOp(graph, dict(min_size=10, framework='tensorflow', box_coordinate_scale=10,
                                             box_size_scale=5, post_nms_topn=max_proposals, feat_stride=feat_stride,
                                             ratio=proposal_ratios, scale=proposal_scales, base_size=anchor_base_size,
                                             pre_nms_topn=2**31 - 1,
                                             nms_thresh=nms_threshold))
        proposal_node = proposal_op.create_node([reshape_permute_node,
                                                 match.single_input_node(0)[0].in_node(0).in_node(0),
                                                 const_node],
                                                dict(name=proposal_op.attrs['type'] + '_'))
```

The box coordinates in the TensorFlow are in the following layout "YXYX" while Inference Engine uses "XYXY" layout so it is necessary to swap coordinates produced by Proposal layer. It is implemented with help of a convolution node with a special filter of a size [5, 5]:

```python
        proposal_reshape_4d_op = Reshape(graph, {'dim': np.array([max_proposals, 1, 1, 5])})
        proposal_reshape_4d_node = proposal_reshape_4d_op.create_node([proposal_node], dict(name="reshape_4d_"))
        update_attrs(proposal_reshape_4d_node, 'shape_attrs', 'dim')

        # create convolution node to swap X and Y coordinates in the proposals
        conv_filter_const_data = np.array(np.array([[1, 0, 0, 0, 0],
                                                    [0, 0, 1, 0, 0],
                                                    [0, 1, 0, 0, 0],
                                                    [0, 0, 0, 0, 1],
                                                    [0, 0, 0, 1, 0]],
                                                   dtype=np.float32).reshape([1, 1, 5, 5]), dtype=np.float32)
        conv_filter_const_op = Const(graph, dict(value=conv_filter_const_data, spatial_dims=np.array([2, 3])))
        conv_filter_const_node = conv_filter_const_op.create_node([], dict(name="conv_weights"))

        conv_op = Op(graph, {
                        'op': 'Conv2D',
                        'bias_addable': False,
                        'spatial_dims': np.array([1, 2]),
                        'channel_dims': np.array([3]),
                        'batch_dims': np.array([0]),
                        'pad': None,
                        'pad_spatial_shape': None,
                        'input_feature_channel': 2,
                        'output_feature_channel': 2,
                        'output_shape': [max_proposals, 1, 1, 5],
                        'dilation': np.array([1, 1, 1, 1], dtype=np.int64),
                        'stride': np.array([1, 1, 1, 1]),
                        'type': 'Convolution',
                        'group': None,
                        'layout': 'NHWC',
                        'infer': __class__.fake_conv_shape_infer})
        predictions_node = conv_op.create_node([proposal_reshape_4d_node, conv_filter_const_node], dict(name="conv_"))
        update_ie_fields(graph.node[predictions_node.id])

        proposal_reshape_2d_op = Reshape(graph, {'dim': np.array([max_proposals, 5])})
        proposal_reshape_2d_node = proposal_reshape_2d_op.create_node([predictions_node], dict(name="reshape_2d_"))
        # set specific name for this Reshape operation so we can use it in the DetectionOutput replacer
        proposal_reshape_2d_node['name'] = 'swapped_proposals'
```

The ROIPooling layer in TensorFlow is implemented with operation called `CropAndResize` with bi-linear filtration. Inference Engine implementation of the ROIPooling layer with bi-linear filtration requires input boxes coordinates be scaled to [0, 1] interval. Adding elementwise multiplication of box coordinates solves this issue:

```python
        # the TF implementation of Proposal with bi-linear filtration need proposals scaled by image size
        proposal_scale_const = np.array([1.0, 1 / input_height, 1 / input_width, 1 / input_height, 1 / input_width],
                                        dtype=np.float32)
        proposal_scale_const_op = Const(graph, dict(value=proposal_scale_const, shape=proposal_scale_const.shape))
        proposal_scale_const_node = proposal_scale_const_op.create_node([], dict(name='Proposal_scale_const_'))

        scale_proposals_op = Eltwise(graph, {'operation': 'mul'})
        scale_proposals_node = scale_proposals_op.create_node([proposal_reshape_2d_node, proposal_scale_const_node],
                                                              dict(name='scale_proposals_'))
```

The last step is to create the ROIPooling node with 2 inputs: the identified feature maps from the `FirstStageFeatureExtractor` and the scaled output of the Proposal layer:

```python
        feature_extractor_output_nodes = scope_output_nodes(graph, 'FirstStageFeatureExtractor')
        if len(feature_extractor_output_nodes) != 1:
            raise Error("Failed to determine FirstStageFeatureExtractor output node to connect it to the ROIPooling."
                        "Found the following nodes: {}".format([node.name for node in feature_extractor_output_nodes]))

        roi_pooling_op = ROIPooling(graph, dict(method="bilinear", framework="tensorflow",
                                                pooled_h=roi_pool_size, pooled_w=roi_pool_size,
                                                spatial_scale=roi_spatial_scale))
        roi_pooling_node = roi_pooling_op.create_node([feature_extractor_output_nodes[0], scale_proposals_node],
                                                      dict(name='ROI_Pooling_'))

        return {'roi_pooling_node': roi_pooling_node}
```

The are two additional methods implemented in the replacer class:

*   The `fake_conv_shape_infer` is a silly infer function for the convolution that permutes X and Y coordinates of the Proposal output which avoids setting a lot of internal attributes required for propoper shape inference.

*   The "classes_probabilities_reshape_shape_infer" function is used to update the output dimension of the reshape operation. The output spatial dimensions depends on the convolution output spatial dimensions thus they are not known until the shape inference pass which is performed after this sub-graph replacement class. So this custom infer function is called instead of default Reshape shape inference function, updates the required attribute "dim" of the node with the convolution output spatial dimensions which are known at the time of calling this inference function and then call the default Reshape inference function.

```python
    @staticmethod
    def fake_conv_shape_infer(node: Node):
        node.out_node(0).shape = node.in_node(0).shape
        # call functions to update internal attributes required for correct IR generation
        mark_input_bins(node)
        assign_dims_to_weights(node.in_node(1), [0, 1], node.input_feature_channel, node.output_feature_channel, 4)

    @staticmethod
    def classes_probabilities_reshape_shape_infer(node: Node):
        # now we can determine the reshape dimensions from Convolution node
        conv_node = node.conv_node
        conv_output_shape = conv_node.out_node().shape

        # update desired shape of the Reshape node
        node.dim = np.array([0, conv_output_shape[1], conv_output_shape[2], node.anchors_count * 2])
        node.old_infer(node)
```

The second replacer defined in the sub-graph replacement configuration file replaces the `SecondStagePostprocessor` block and is defined using scope:

```json
    {
        "custom_attributes": {
            "code_type": "caffe.PriorBoxParameter.CENTER_SIZE",
            "confidence_threshold": 0.01,
            "keep_top_k": 300,
            "nms_threshold": 0.6,
            "pad_mode": "caffe.ResizeParameter.CONSTANT",
            "resize_mode": "caffe.ResizeParameter.WARP",
            "max_detections_per_class": 100,
            "num_classes": 90
        },
        "id": "SecondStagePostprocessorReplacement",
        "inputs": [
            [
                {
                    "node": "Reshape$",
                    "port": 0
                }
            ],
            [
                {
                    "node": "Reshape_1$",
                    "port": 0
                }
            ],
            [
                {
                    "node": "ExpandDims$",
                    "port": 0
                }
            ]
        ],
        "instances": [
            ".*SecondStagePostprocessor/"
        ],
        "match_kind": "scope",
        "outputs": [
            {
                "node": "BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArrayGatherV3$",
                "port": 0
            }
        ]
    }
```

The replacement code is similar to the `SecondStagePostprocessor` replacement for the SSDs topologies. The are two major difference:

*   The tensor with bounding boxes doesn't contain locations for class 0 (background class) but Inference Engine Detection Output layer requires it. The Const node with some dummy values are created and concatenated with the tensor.

*   The priors tensor is not constant like in SSDs so the bounding boxes tensor must be scaled with variances [0.1, 0.1, 0.2, 0.2].

The descibed above difference are resolved with the following code:

```python
        # TF produces locations tensor without boxes for background.
        # Inference Engine DetectionOutput layer requires background boxes so we generate them with some values
        # and concatenate with locations tensor
        fake_background_locs_blob = np.tile([[[1, 1, 2, 2]]], [max_detections_per_class, 1, 1])
        fake_background_locs_const_op = Const(graph, dict(value=fake_background_locs_blob,
                                                          shape=fake_background_locs_blob.shape))
        fake_background_locs_const_node = fake_background_locs_const_op.create_node([])

        reshape_loc_op = Reshape(graph, {'dim': np.array([max_detections_per_class, num_classes, 4])})
        reshape_loc_node = reshape_loc_op.create_node([match.single_input_node(0)[0].in_node(0)],
                                                      dict(name='Reshape_loc_'))

        concat_loc_op = Concat(graph, {'axis': 1})
        concat_loc_node = concat_loc_op.create_node([fake_background_locs_const_node, reshape_loc_node],
                                                    dict(name='Concat_fake_loc_'))

        # blob with variances
        variances_blob = np.array([0.1, 0.1, 0.2, 0.2])
        variances_const_op = Const(graph, dict(value=variances_blob, shape=variances_blob.shape))
        variances_const_node = variances_const_op.create_node([])

        # reshape locations tensor to 2D so it could be passed to Eltwise which will be converted to ScaleShift
        reshape_loc_2d_op = Reshape(graph, {'dim': np.array([-1, 4])})
        reshape_loc_2d_node = reshape_loc_2d_op.create_node([concat_loc_node], dict(name='reshape_locs_2d_'))

        # element-wise multiply locations with variances
        eltwise_locs_op = Eltwise(graph, {'operation': 'mul'})
        eltwise_locs_node = eltwise_locs_op.create_node([reshape_loc_2d_node, variances_const_node],
                                                        dict(name='scale_locs_'))
```

### Example of Model Optimizer Command-Line for TensorFlow's Faster R-CNNs
The final command line to convert Faster R-CNNs from the TensorFlow* Object Detection Zoo is the following:

```sh
./mo.py --input_model=<path_to_frozen.pb> --output=detection_boxes,detection_scores,num_detections --tensorflow_use_custom_operations_config extensions/front/tf/legacy_faster_rcnn_support.json
```

Note that there are minor changes that should be made to the and sub-graph replacement configuration file `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/legacy_faster_rcnn_support.json` before converting particular Faster R-CNN topology. Refer to the table below.

### Sub-Graph Replacement Configuration File Parameters to Convert Different Faster R-CNN Models
|Model Name | Configuration File Changes|
|:----|:----:|
| faster_rcnn_inception_v2_coco | None
| faster_rcnn_resnet50_coco | None
| faster_rcnn_resnet50_lowproposals_coco | None
| faster_rcnn_resnet101_coco | None
| faster_rcnn_resnet101_lowproposals_coco | None
| faster_rcnn_inception_resnet_v2_atrous_coco | "feat_stride: 8"
| faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco| "feat_stride: 8"

