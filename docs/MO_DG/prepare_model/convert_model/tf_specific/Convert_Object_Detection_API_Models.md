# Converting TensorFlow* Object Detection API Models {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models}

> **NOTES**:
> * Starting with the 2022.1 release, the Model Optimizer can convert the TensorFlow\* Object Detection API Faster and Mask RCNNs topologies differently. By default, the Model Optimizer adds operation "Proposal" to the generated IR. This operation needs an additional input to the model with name "image_info" which should be fed with several values describing the pre-processing applied to the input image (refer to the [Proposal](../../../../ops/detection/Proposal_4.md) operation specification for more information). However, this input is redundant for the models trained and inferred with equal size images. Model Optimizer can generate IR for such models and insert operation [DetectionOutput](../../../../ops/detection/DetectionOutput_1.md) instead of `Proposal`. The `DetectionOutput` operation does not require additional model input "image_info" and moreover, for some models the produced inference results are closer to the original TensorFlow\* model. In order to trigger new behaviour the attribute "operation_to_add" in the corresponding JSON transformation configuration file should be set to value "DetectionOutput" instead of default one "Proposal".
> * Starting with the 2021.1 release, the Model Optimizer converts the TensorFlow\* Object Detection API SSDs, Faster and Mask RCNNs topologies keeping shape-calculating sub-graphs by default, so topologies can be re-shaped in the Inference Engine using dedicated reshape API. Refer to [Using Shape Inference](../../../../IE_DG/ShapeInference.md) for more information on how to use this feature. It is possible to change the both spatial dimensions of the input image and batch size.
> * To generate IRs for TF 1 SSD topologies, the Model Optimizer creates a number of `PriorBoxClustered` operations instead of a constant node with prior boxes calculated for the particular input image size. This change allows you to reshape the topology in the Inference Engine using dedicated Inference Engine API. The reshaping is supported for all SSD topologies except FPNs which contain hardcoded shapes for some operations preventing from changing topology input shape.

## How to Convert a Model

You can download TensorFlow\* Object Detection API models from the <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md">TensorFlow 1 Detection Model Zoo</a> or <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">TensorFlow 2 Detection Model Zoo</a>.

<strong>NOTE</strong>: Before converting, make sure you have configured the Model Optimizer. For configuration steps, refer to [Configuring the Model Optimizer](../../Config_Model_Optimizer.md).

To convert a TensorFlow\* Object Detection API model, go to the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory and run the `mo_tf.py` script with the following required parameters:

* `--input_model <path_to_frozen.pb>` --- File with a pre-trained model (binary or text .pb file after freezing) OR `--saved_model_dir <path_to_saved_model>` for the TensorFlow\* 2 models
* `--transformations_config <path_to_subgraph_replacement_configuration_file.json>` --- A subgraph replacement configuration file with transformations description. For the models downloaded from the TensorFlow\* Object Detection API zoo, you can find the configuration files in the `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf` directory. Use:
    * `ssd_v2_support.json` --- for frozen SSD topologies from the models zoo version up to 1.13.X inclusively
    * `ssd_support_api_v.1.14.json` --- for SSD topologies trained using the TensorFlow\* Object Detection API version 1.14 up to 1.14.X inclusively
    * `ssd_support_api_v.1.15.json` --- for SSD topologies trained using the TensorFlow\* Object Detection API version 1.15 up to 2.0
    * `ssd_support_api_v.2.0.json` --- for SSD topologies trained using the TensorFlow\* Object Detection API version 2.0 up to 2.3.X inclusively
    * `ssd_support_api_v.2.4.json` --- for SSD topologies trained using the TensorFlow\* Object Detection API version 2.4 or higher
    * `efficient_det_support_api_v.2.0.json` --- for EfficientDet topologies trained using the TensorFlow\* Object Detection API version 2.0 up to 2.3.X inclusively
    * `efficient_det_support_api_v.2.4.json` --- for EfficientDet topologies trained using the TensorFlow\* Object Detection API version 2.4 or higher
    * `faster_rcnn_support.json` --- for Faster R-CNN topologies from the TF 1.X models zoo trained with TensorFlow\* version up to 1.6.X inclusively
    * `faster_rcnn_support_api_v1.7.json` --- for Faster R-CNN topologies trained using the TensorFlow\* Object Detection API version 1.7.0 up to 1.9.X inclusively
    * `faster_rcnn_support_api_v1.10.json` --- for Faster R-CNN topologies trained using the TensorFlow\* Object Detection API version 1.10.0 up to 1.12.X inclusively
    * `faster_rcnn_support_api_v1.13.json` --- for Faster R-CNN topologies trained using the TensorFlow\* Object Detection API version 1.13.X
    * `faster_rcnn_support_api_v1.14.json` --- for Faster R-CNN topologies trained using the TensorFlow\* Object Detection API version 1.14.0 up to 1.14.X inclusively
    * `faster_rcnn_support_api_v1.15.json` --- for Faster R-CNN topologies trained using the TensorFlow\* Object Detection API version 1.15.0 up to 2.0
    * `faster_rcnn_support_api_v2.0.json` --- for Faster R-CNN topologies trained using the TensorFlow\* Object Detection API version 2.0 up to 2.3.X inclusively
    * `faster_rcnn_support_api_v2.4.json` --- for Faster R-CNN topologies trained using the TensorFlow\* Object Detection API version 2.4 or higher
    * `mask_rcnn_support.json` --- for Mask R-CNN topologies from the TF 1.X models zoo trained with TensorFlow\* version 1.9.0 or lower.
    * `mask_rcnn_support_api_v1.7.json` --- for Mask R-CNN topologies trained using the TensorFlow\* Object Detection API version 1.7.0 up to 1.9.X inclusively
    * `mask_rcnn_support_api_v1.11.json` --- for Mask R-CNN topologies trained using the TensorFlow\* Object Detection API version 1.11.0 up to 1.12.X inclusively
    * `mask_rcnn_support_api_v1.13.json` --- for Mask R-CNN topologies trained using the TensorFlow\* Object Detection API version 1.13.0 up to 1.13.X inclusively
    * `mask_rcnn_support_api_v1.14.json` --- for Mask R-CNN topologies trained using the TensorFlow\* Object Detection API version 1.14.0 up to 1.14.X inclusively
    * `mask_rcnn_support_api_v1.15.json` --- for Mask R-CNN topologies trained using the TensorFlow\* Object Detection API version 1.15.0 up to 2.0
    * `mask_rcnn_support_api_v2.0.json` --- for Mask R-CNN topologies trained using the TensorFlow\* Object Detection API version 2.0 up to 2.3.X inclusively
    * `mask_rcnn_support_api_v2.4.json` --- for Mask R-CNN topologies trained using the TensorFlow\* Object Detection API version 2.4 or higher
    * `rfcn_support.json` --- for RFCN topology from the models zoo trained with TensorFlow\* version up to 1.9.X inclusively
    * `rfcn_support_api_v1.10.json` --- for RFCN topology from the models zoo frozen with TensorFlow\* version 1.10.0 up to 1.12.X inclusively
    * `rfcn_support_api_v1.13.json` --- for RFCN topology from the models zoo frozen with TensorFlow\* version 1.13.X
    * `rfcn_support_api_v1.14.json` --- for RFCN topology from the models zoo frozen with TensorFlow\* version 1.14.0 or higher
* `--tensorflow_object_detection_api_pipeline_config <path_to_pipeline.config>` --- A special configuration file that describes the topology hyper-parameters and structure of the TensorFlow Object Detection API model. For the models downloaded from the TensorFlow\* Object Detection API zoo, the configuration file is named `pipeline.config`. If you plan to train a model yourself, you can find templates for these files in the [models repository](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).
* `--input_shape` (optional) --- A custom input image shape. Refer to [Custom Input Shape](#tf_od_custom_input_shape) for more information how the `--input_shape` parameter is handled for the TensorFlow* Object Detection API models.

> **NOTE:** The color channel order (RGB or BGR) of an input data should match the channel order of the model training dataset. If they are different, perform the `RGB<->BGR` conversion specifying the command-line parameter: `--reverse_input_channels`. Otherwise, inference results may be incorrect. If you convert a TensorFlow\* Object Detection API model to use with the Inference Engine sample applications, you must specify the `--reverse_input_channels` parameter. For more information about the parameter, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../Converting_Model_General.md).

Additionally to the mandatory parameters listed above you can use optional conversion parameters if needed. A full list of parameters is available in the [Converting a TensorFlow* Model](../Convert_Model_From_TensorFlow.md) topic.

For example, if you downloaded the [pre-trained SSD InceptionV2 topology](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) and extracted archive to the directory `/tmp/ssd_inception_v2_coco_2018_01_28`, the sample command line to convert the model looks as follows:

```
<INSTALL_DIR>/deployment_tools/model_optimizer/mo_tf.py --input_model=/tmp/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --transformations_config <INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /tmp/ssd_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels
```

## Important Notes About Feeding Input Images to the Samples

Inference Engine comes with a number of samples to infer Object Detection API models including:

* [Object Detection for SSD Sample](../../../../../inference-engine/samples/object_detection_sample_ssd/README.md) --- for RFCN, SSD and Faster R-CNNs
* [Mask R-CNN Sample for TensorFlow* Object Detection API Models](@ref omz_demos_mask_rcnn_demo_cpp) --- for Mask R-CNNs

There are several important notes about feeding input images to the samples:

1. Inference Engine samples stretch input image to the size of the input operation without preserving aspect ratio. This behavior is usually correct for most topologies (including SSDs), but incorrect for other models like Faster R-CNN, Mask R-CNN and R-FCN. These models usually use keeps aspect ratio resizer. The type of pre-processing is defined in the pipeline configuration file in the section `image_resizer`. If keeping aspect ratio is used, then it is necessary to resize image before passing it to the sample and optionally pad the resized image with 0s (if the attribute "pad_to_max_dimension" in the pipeline.config is equal to "true").

2. TensorFlow\* implementation of image resize may be different from the one implemented in the sample. Even reading input image from compressed format (like `.jpg`) could give different results in the sample and TensorFlow\*. So, if it is necessary to compare accuracy between the TensorFlow\* and the Inference Engine it is recommended to pass pre-resized input image in a non-compressed format (like `.bmp`).

3. If you want to infer the model with the Inference Engine samples, convert the model specifying the `--reverse_input_channels` command line parameter. The samples load images in BGR channels order, while TensorFlow* models were trained with images in RGB order. When the `--reverse_input_channels` command line parameter is specified, the Model Optimizer performs first convolution or other channel dependent operation weights modification so the output will be like the image is passed with RGB channels order.

4. Read carefully messaged printed by the Model Optimizer during a model conversion. They contain important instructions on how to prepare input data before running the inference and how to interpret the output.

## Custom Input Shape <a name="tf_od_custom_input_shape"></a>
Model Optimizer handles the command line parameter `--input_shape` for TensorFlow\* Object Detection API models in a special way depending on the image resizer type defined in the `pipeline.config` file. TensorFlow\* Object Detection API generates different `Preprocessor` sub-graph based on the image resizer type. Model Optimizer supports two types of image resizer:
* `fixed_shape_resizer` --- *Stretches* input image to the specific height and width. The `pipeline.config` snippet below shows a `fixed_shape_resizer` sample definition:
```
image_resizer {
  fixed_shape_resizer {
    height: 300
    width: 300
  }
}
```
* `keep_aspect_ratio_resizer` --- Resizes the input image *keeping aspect ratio* to satisfy the minimum and maximum size constraints. The `pipeline.config` snippet below shows a `keep_aspect_ratio_resizer` sample definition:
```
image_resizer {
  keep_aspect_ratio_resizer {
    min_dimension: 600
    max_dimension: 1024
  }
}
```
If an additional parameter "pad_to_max_dimension" is equal to "true" then the resized image will be padded with 0s to the square image of size "max_dimension".

### Fixed Shape Resizer Replacement
* If the `--input_shape` command line parameter is not specified, the Model Optimizer generates an input operation with the height and width as defined in the `pipeline.config`.

* If the `--input_shape [1, H, W, 3]` command line parameter is specified, the Model Optimizer sets the input operation height to `H` and width to `W` and convert the model. However, the conversion may fail because of the following reasons:
  * The model is not reshape-able, meaning that it's not possible to change the size of the model input image. For example, SSD FPN models have `Reshape` operations with hard-coded output shapes, but the input size to these `Reshape` instances depends on the input image size. In this case, the Model Optimizer shows an error during the shape inference phase. Run the Model Optimizer with `--log_level DEBUG` to see the inferred operations output shapes to see the mismatch.
  * Custom input shape is too small. For example, if you specify `--input_shape [1,100,100,3]` to convert a SSD Inception V2 model, one of convolution or pooling nodes decreases input tensor spatial dimensions to non-positive values. In this case, the Model Optimizer shows error message like this: '[ ERROR ]  Shape [  1  -1  -1 256] is not fully defined for output X of "node_name".'


### Keep Aspect Ratio Resizer Replacement
* If the `--input_shape` command line parameter is not specified, the Model Optimizer generates an input operation with both height and width equal to the value of parameter `min_dimension` in the `keep_aspect_ratio_resizer`.

* If the `--input_shape [1, H, W, 3]` command line parameter is specified, the Model Optimizer scales the specified input image height `H` and width `W` to satisfy the `min_dimension` and `max_dimension` constraints defined in the `keep_aspect_ratio_resizer`. The following function calculates the input operation height and width:

```python
def calculate_shape_keeping_aspect_ratio(H: int, W: int, min_dimension: int, max_dimension: int):
    ratio_min = min_dimension / min(H, W)
    ratio_max = max_dimension / max(H, W)
    ratio = min(ratio_min, ratio_max)
    return int(round(H * ratio)), int(round(W * ratio))
```
The `--input_shape` command line parameter should be specified only if the "pad_to_max_dimension" does not exist of is set to "false" in the `keep_aspect_ratio_resizer`.

Models with `keep_aspect_ratio_resizer` were trained to recognize object in real aspect ratio, in contrast with most of the classification topologies trained to recognize objects stretched vertically and horizontally as well. By default, the Model Optimizer converts topologies with `keep_aspect_ratio_resizer` to consume a square input image. If the non-square image is provided as input, it is stretched without keeping aspect ratio that results to object detection quality decrease.

> **NOTE**: It is highly recommended specifying the `--input_shape` command line parameter for the models with `keep_aspect_ratio_resizer` if the input image dimensions are known in advance.

## Detailed Explanations of Model Conversion Process

This section is intended for users who want to understand how the Model Optimizer performs Object Detection API models conversion in details. The knowledge given in this section is also useful for users having complex models that are not converted with the Model Optimizer out of the box. It is highly recommended to read [Sub-Graph Replacement in Model Optimizer](../../customize_model_optimizer/Subgraph_Replacement_Model_Optimizer.md) chapter first to understand sub-graph replacement concepts which are used here.

It is also important to open the model in the [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) to see the topology structure. Model Optimizer can create an event file that can be then fed to the TensorBoard* tool. Run the Model Optimizer with providing two command line parameters:
* `--input_model <path_to_frozen.pb>` --- Path to the frozen model
* `--tensorboard_logdir` --- Path to the directory where TensorBoard looks for the event files.

Implementation of the transformations for Object Detection API models is located in the file `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/ObjectDetectionAPI.py`. Refer to the code in this file to understand the details of the conversion process.

### Faster R-CNN Topologies
The Faster R-CNN models contain several building blocks similar to building blocks from SSD models so it is highly recommended to read the section about converting them first. Detailed information about Faster R-CNN topologies is provided [in the abstract](https://arxiv.org/abs/1506.01497).

#### Proposal Layer
The `Proposal` layer is implemented with dozens of primitive operations in TensorFlow, meanwhile, it is a single layer in the Inference Engine. The `ObjectDetectionAPIProposalReplacement` sub-graph replacer identifies nodes corresponding to the layer and replaces them with required new nodes.

```python
class ObjectDetectionAPIProposalReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    This class replaces sub-graph of operations with Proposal layer and additional layers transforming
    tensors from layout of TensorFlow to layout required by Inference Engine.
    Refer to comments inside the function for more information about performed actions.
    """
    replacement_id = 'ObjectDetectionAPIProposalReplacement'

    def run_after(self):
        return [ObjectDetectionAPIPreprocessorReplacement]

    def run_before(self):
        return [Sub, CropAndResizeReplacement]

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['proposal_node'].id}

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        new_list = match.matched_nodes_names().copy()
        # do not remove nodes that produce box predictions and class predictions
        new_list.remove(match.single_input_node(0)[0].id)
        new_list.remove(match.single_input_node(1)[0].id)
        return new_list

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        max_proposals = _value_or_raise(match, pipeline_config, 'first_stage_max_proposals')
        proposal_ratios = _value_or_raise(match, pipeline_config, 'anchor_generator_aspect_ratios')
        proposal_scales = _value_or_raise(match, pipeline_config, 'anchor_generator_scales')
        anchors_count = len(proposal_ratios) * len(proposal_scales)

        # Convolution/matmul node that produces classes predictions
        # Permute result of the tensor with classes permissions so it will be in a correct layout for Softmax
        predictions_node = backward_bfs_for_operation(match.single_input_node(1)[0], ['Add'])[0]

        reshape_classes_op = Reshape(graph, dict(dim=int64_array([0, anchors_count, 2, -1])))
        reshape_classes_node = reshape_classes_op.create_node([], dict(name='predictions/Reshape', nchw_layout=True))
        predictions_node.insert_node_after(reshape_classes_node, 0)

        softmax_conf_op = Softmax(graph, dict(axis=2, nchw_layout=True, name=reshape_classes_node.id + '/Softmax'))
        softmax_conf_node = softmax_conf_op.create_node([reshape_classes_node])
        permute_reshape_softmax_op = Permute(graph, dict(order=int64_array([0, 2, 1, 3]), nchw_layout=True))
        permute_reshape_softmax_node = permute_reshape_softmax_op.create_node([softmax_conf_node], dict(
            name=softmax_conf_node.name + '/Permute'))

        initial_shape_op = Shape(graph, dict(name=predictions_node.id + '/Shape'))
        initial_shape_node = initial_shape_op.create_node([predictions_node])

        # implement custom reshape infer function because we need to know the input convolution node output dimension
        # sizes but we can know it only after partial infer
        reshape_permute_op = Reshape(graph, dict())
        reshape_permute_node = reshape_permute_op.create_node([permute_reshape_softmax_node, initial_shape_node],
                                                              dict(name='Reshape_Permute_Class'))

        variance_height = pipeline_config.get_param('frcnn_variance_height')
        variance_width = pipeline_config.get_param('frcnn_variance_width')
        variance_x = pipeline_config.get_param('frcnn_variance_x')
        variance_y = pipeline_config.get_param('frcnn_variance_y')
        anchor_generator_height_stride = pipeline_config.get_param('anchor_generator_height_stride')
        anchor_generator_width_stride = pipeline_config.get_param('anchor_generator_width_stride')
        anchor_generator_height = pipeline_config.get_param('anchor_generator_height')
        anchor_generator_width = pipeline_config.get_param('anchor_generator_width')

        if variance_height != variance_width:
            log.error('The values for variance for height "{}" is not equal to variance for width "{}". The detection '
                      'results will be inaccurate.'.format(variance_height, variance_width))
        if variance_x != variance_y:
            log.error('The values for variance for x "{}" is not equal to variance for y "{}". The detection '
                      'results will be inaccurate.'.format(variance_x, variance_y))
        if anchor_generator_height_stride != anchor_generator_width_stride:
            log.error('The values for the anchor generator height stride "{}" is not equal to the anchor generator '
                      'width stride "{}". The detection results will be inaccurate.'.format(
                anchor_generator_height_stride, anchor_generator_width_stride))
        if anchor_generator_height != anchor_generator_width:
            log.error('The values for the anchor generator height "{}" is not equal to the anchor generator width '
                      'stride "{}". The detection results will be inaccurate.'.format(anchor_generator_height,
                                                                                      anchor_generator_width))

        proposal_op = ProposalOp(graph, dict(min_size=1,
                                             framework='tensorflow',
                                             pre_nms_topn=2 ** 31 - 1,
                                             box_size_scale=variance_height,
                                             box_coordinate_scale=variance_x,
                                             post_nms_topn=max_proposals,
                                             feat_stride=anchor_generator_height_stride,
                                             ratio=proposal_ratios,
                                             scale=proposal_scales,
                                             normalize=1,
                                             base_size=anchor_generator_height,
                                             nms_thresh=_value_or_raise(match, pipeline_config,
                                                                        'first_stage_nms_iou_threshold')))
        for key in ('clip_before_nms', 'clip_after_nms'):
            if key in match.custom_replacement_desc.custom_attributes:
                proposal_op.attrs[key] = int(match.custom_replacement_desc.custom_attributes[key])

        anchors_node = backward_bfs_for_operation(match.single_input_node(0)[0], ['Add'])[0]

        # creates input to store input image height, width and scales (usually 1.0s)
        # the batch size for this input is fixed because it is allowed to pass images of the same size only as input
        input_op_with_image_size = Input(graph, dict(shape=int64_array([1, 3]), fixed_batch=True))
        input_with_image_size_node = input_op_with_image_size.create_node([], dict(name='image_info'))

        proposal_node = proposal_op.create_node([reshape_permute_node, anchors_node, input_with_image_size_node],
                                                dict(name='proposals'))

        if 'do_not_swap_proposals' in match.custom_replacement_desc.custom_attributes and \
                match.custom_replacement_desc.custom_attributes['do_not_swap_proposals']:
            swapped_proposals_node = proposal_node
        else:
            swapped_proposals_node = add_convolution_to_swap_xy_coordinates(graph, proposal_node, 5)

        proposal_reshape_2d_op = Reshape(graph, dict(dim=int64_array([-1, 5]), nchw_layout=True))
        proposal_reshape_2d_node = proposal_reshape_2d_op.create_node([swapped_proposals_node],
                                                                      dict(name="reshape_swap_proposals_2d"))

        # feed the CropAndResize node with a correct boxes information produced with the Proposal layer
        # find the first CropAndResize node in the BFS order
        crop_and_resize_nodes_ids = [node_id for node_id in bfs_search(graph, [match.single_input_node(0)[0].id]) if
                                     graph.node[node_id]['op'] == 'CropAndResize']
        assert len(crop_and_resize_nodes_ids) != 0, "Didn't find any CropAndResize nodes in the graph."
        if 'do_not_swap_proposals' not in match.custom_replacement_desc.custom_attributes or not \
                match.custom_replacement_desc.custom_attributes['do_not_swap_proposals']:
            crop_and_resize_node = Node(graph, crop_and_resize_nodes_ids[0])
            # set a marker that the input with box coordinates has been pre-processed so the CropAndResizeReplacement
            # transform doesn't try to merge the second and the third inputs
            crop_and_resize_node['inputs_preprocessed'] = True
            graph.remove_edge(crop_and_resize_node.in_node(1).id, crop_and_resize_node.id)
            graph.create_edge(proposal_reshape_2d_node, crop_and_resize_node, out_port=0, in_port=1)

        tf_proposal_reshape_4d_op = Reshape(graph, dict(dim=int64_array([-1, 1, max_proposals, 5]), nchw_layout=True))
        tf_proposal_reshape_4d_node = tf_proposal_reshape_4d_op.create_node([swapped_proposals_node],
                                                                            dict(name="reshape_proposal_4d"))

        crop_op = Crop(graph, dict(axis=int64_array([3]), offset=int64_array([1]), dim=int64_array([4]),
                                   nchw_layout=True))
        crop_node = crop_op.create_node([tf_proposal_reshape_4d_node], dict(name='crop_proposals'))

        tf_proposals_crop_reshape_3d_op = Reshape(graph, dict(dim=int64_array([0, -1, 4]), nchw_layout=True))
        tf_proposals_crop_reshape_3d_node = tf_proposals_crop_reshape_3d_op.create_node([crop_node],
                                                                                        dict(name="reshape_crop_3d"))

        return {'proposal_node': tf_proposals_crop_reshape_3d_node}
```
The main interest of the implementation of this replacer is the `generate_sub_graph` function.

Lines 26-34: Parses the `pipeline.config` file and gets required parameters for the `Proposal` layer.

Lines 38-57: Performs the following manipulations with the tensor with class predictions. TensorFlow uses the NHWC layout, while the Inference Engine uses NCHW. Model Optimizer by default performs transformations with all nodes data in the inference graph to convert it to the NCHW layout. The size of 'C' dimension of the tensor with class predictions is equal to \f$base\_anchors\_count \cdot 2\f$, where 2 corresponds to a number of classes (background and foreground) and \f$base\_anchors\_count\f$ is equal to number of anchors that are applied to each position of 'H' and 'W' dimensions. Therefore, there are \f$H \cdot W \cdot base\_anchors\_count\f$ bounding boxes. Lines 44-45 apply the Softmax layer to this tensor to get class probabilities for each bounding box.

Lines 59-81: Reads topology parameters related to variances and anchors generation. 

Lines 83-108: Adds the `Proposal` layer to the graph. This layer has one input (generated in lines 104-105) which should be filled with three values before inference: input image height, input image width, image scale factor.

Lines 110-132: Swaps output values of the `Proposal` layer if the parameter `do_not_swap_proposals` is not set to `True` in the sub-graph replacement configuration file for the replacer. 

Lines 134-144: Crops the output from the `Proposal` node to remove the batch indices (the Inference Engine implementation of the `Proposal` layer generates tensor with shape `[num_proposals, 5]`). The final tensor contains just box coordinates as in the TensorFlow implementation.

#### SecondStagePostprocessor Block
The `SecondStagePostprocessor` block is similar to the `Postprocessor` block from the SSDs topologies. But there are a number of differences in conversion of the `SecondStagePostprocessor` block.

```python
class ObjectDetectionAPIDetectionOutputReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    Replaces the sub-graph that is equal to the DetectionOutput layer from Inference Engine. This replacer is used for
    Faster R-CNN, R-FCN and Mask R-CNN topologies conversion.
    The replacer uses a value of the custom attribute 'coordinates_swap_method' from the sub-graph replacement
    configuration file to choose how to swap box coordinates of the 0-th input of the generated DetectionOutput layer.
    Refer to the code for more details.
    """
    replacement_id = 'ObjectDetectionAPIDetectionOutputReplacement'

    def run_before(self):
        return [ObjectDetectionAPIMaskRCNNROIPoolingSecondReplacement, Unpack, Sub]

    def run_after(self):
        return [ObjectDetectionAPIProposalReplacement, CropAndResizeReplacement]

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names().copy()
        outputs = ['detection_boxes', 'detection_scores', 'num_detections']
        for output in outputs:
            children = Node(graph, output).out_nodes()
            if len(children) != 1:
                log.warning('Output {} has {} children. It should have only one output: with op==`OpOutput`'
                            ''.format(output, len(children)))
            elif children[list(children.keys())[0]].op == 'OpOutput':
                new_nodes_to_remove.append(children[list(children.keys())[0]].id)
            else:
                continue
        new_nodes_to_remove.extend(outputs)
        return new_nodes_to_remove

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        # the DetectionOutput in IE produces single tensor, but in TF it produces four tensors, so we need to create
        # only one output edge match
        return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}

    @staticmethod
    def skip_nodes_by_condition(current_node: Node, condition: callable):
        while condition(current_node):
            current_node = current_node.in_node()
        return current_node

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)

        num_classes = _value_or_raise(match, pipeline_config, 'num_classes')
        max_proposals = _value_or_raise(match, pipeline_config, 'first_stage_max_proposals')
        activation_function = _value_or_raise(match, pipeline_config, 'postprocessing_score_converter')

        activation_conf_node = add_activation_function_after_node(graph, match.single_input_node(1)[0].in_node(0),
                                                                  activation_function)

        # IE DetectionOutput layer consumes flattened tensors so need add a Reshape layer.
        # The batch value of the input tensor is not equal to the batch of the topology, so it is not possible to use
        # "0" value in the Reshape layer attribute to refer to the batch size, but we know how to
        # calculate the second dimension so the batch value will be deduced from it with help of "-1".
        reshape_conf_op = Reshape(graph, dict(dim=int64_array([-1, (num_classes + 1) * max_proposals])))
        reshape_conf_node = reshape_conf_op.create_node([activation_conf_node], dict(name='do_reshape_conf'))

        # Workaround for PermuteForReshape pass.
        # We looking for first not Reshape-typed node before match.single_input_node(0)[0].in_node(0).
        # And add  reshape_loc node after this first not Reshape-typed node.
        current_node = self.skip_nodes_by_condition(match.single_input_node(0)[0].in_node(0),
                                                    lambda x: x['kind'] == 'op' and x.soft_get('type') == 'Reshape')

        reshape_loc_op = Reshape(graph, dict(dim=int64_array([-1, num_classes, 1, 4])))
        reshape_loc_node = reshape_loc_op.create_node([current_node], dict(name='reshape_loc', nchw_layout=True))
        update_attrs(reshape_loc_node, 'shape_attrs', 'dim')

        # constant node with variances
        variances_const_op = Const(graph, dict(value=_variance_from_pipeline_config(pipeline_config)))
        variances_const_node = variances_const_op.create_node([])

        # TF produces locations tensor without boxes for background.
        # Inference Engine DetectionOutput layer requires background boxes so we generate them
        loc_node = add_fake_background_loc(graph, reshape_loc_node)
        PermuteAttrs.set_permutation(reshape_loc_node, loc_node, None)

        # reshape locations tensor to 2D so it could be passed to Eltwise which will be converted to ScaleShift
        reshape_loc_2d_op = Reshape(graph, dict(dim=int64_array([-1, 4])))
        reshape_loc_2d_node = reshape_loc_2d_op.create_node([loc_node], dict(name='reshape_locs_2d', nchw_layout=True))
        PermuteAttrs.set_permutation(loc_node, reshape_loc_2d_node, None)

        # element-wise multiply locations with variances
        eltwise_locs_op = Eltwise(graph, dict(operation='mul'))
        eltwise_locs_node = eltwise_locs_op.create_node([reshape_loc_2d_node, variances_const_node],
                                                        dict(name='scale_locs'))

        # IE DetectionOutput layer consumes flattened tensors so need add a Reshape layer.
        # The batch value of the input tensor is not equal to the batch of the topology, so it is not possible to use
        # "0" value in the Reshape layer attribute to refer to the batch size, but we know how to
        # calculate the second dimension so the batch value will be deduced from it with help of "-1".
        reshape_loc_do_op = Reshape(graph, dict(dim=int64_array([-1, (num_classes + 1) * max_proposals * 4])))

        custom_attributes = match.custom_replacement_desc.custom_attributes
        coordinates_swap_method = 'add_convolution'
        if 'coordinates_swap_method' not in custom_attributes:
            log.error('The ObjectDetectionAPIDetectionOutputReplacement sub-graph replacement configuration file '
                      'must contain "coordinates_swap_method" in the "custom_attributes" dictionary. Two values are '
                      'supported: "swap_weights" and "add_convolution". The first one should be used when there is '
                      'a MatMul or Conv2D node before the "SecondStagePostprocessor" block in the topology. With this '
                      'solution the weights of the MatMul or Conv2D nodes are permutted, simulating the swap of XY '
                      'coordinates in the tensor. The second could be used in any other cases but it is worse in terms '
                      'of performance because it adds the Conv2D node which performs permutting of data. Since the '
                      'attribute is not defined the second approach is used by default.')
        else:
            coordinates_swap_method = custom_attributes['coordinates_swap_method']
        supported_swap_methods = ['swap_weights', 'add_convolution']
        if coordinates_swap_method not in supported_swap_methods:
            raise Error('Unsupported "coordinates_swap_method" defined in the sub-graph replacement configuration '
                        'file. Supported methods are: {}'.format(', '.join(supported_swap_methods)))

        if coordinates_swap_method == 'add_convolution':
            swapped_locs_node = add_convolution_to_swap_xy_coordinates(graph, eltwise_locs_node, 4)
            reshape_loc_do_node = reshape_loc_do_op.create_node([swapped_locs_node], dict(name='do_reshape_locs'))
        else:
            reshape_loc_do_node = reshape_loc_do_op.create_node([eltwise_locs_node], dict(name='do_reshape_locs'))

        # find Proposal output which has the data layout as in TF: YXYX coordinates without batch indices.
        proposal_nodes_ids = [node_id for node_id, attrs in graph.nodes(data=True)
                              if 'name' in attrs and attrs['name'] == 'crop_proposals']
        if len(proposal_nodes_ids) != 1:
            raise Error("Found the following nodes '{}' with name 'crop_proposals' but there should be exactly 1. "
                        "Looks like ObjectDetectionAPIProposalReplacement replacement didn't work.".
                        format(proposal_nodes_ids))
        proposal_node = Node(graph, proposal_nodes_ids[0])

        # check whether it is necessary to permute proposals coordinates before passing them to the DetectionOutput
        # currently this parameter is set for the RFCN topologies
        if 'swap_proposals' in custom_attributes and custom_attributes['swap_proposals']:
            proposal_node = add_convolution_to_swap_xy_coordinates(graph, proposal_node, 4)

        # reshape priors boxes as Detection Output expects
        reshape_priors_op = Reshape(graph, dict(dim=int64_array([-1, 1, max_proposals * 4])))
        reshape_priors_node = reshape_priors_op.create_node([proposal_node],
                                                            dict(name='DetectionOutput_reshape_priors_'))

        detection_output_op = DetectionOutput(graph, {})
        if coordinates_swap_method == 'swap_weights':
            # update infer function to re-pack weights
            detection_output_op.attrs['old_infer'] = detection_output_op.attrs['infer']
            detection_output_op.attrs['infer'] = __class__.do_infer
        for key in ('clip_before_nms', 'clip_after_nms'):
            if key in match.custom_replacement_desc.custom_attributes:
                detection_output_op.attrs[key] = int(match.custom_replacement_desc.custom_attributes[key])

        detection_output_node = detection_output_op.create_node(
            [reshape_loc_do_node, reshape_conf_node, reshape_priors_node],
            dict(name=detection_output_op.attrs['type'], share_location=0, variance_encoded_in_target=1,
                 code_type='caffe.PriorBoxParameter.CENTER_SIZE', pad_mode='caffe.ResizeParameter.CONSTANT',
                 resize_mode='caffe.ResizeParameter.WARP',
                 num_classes=num_classes,
                 confidence_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_score_threshold'),
                 top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_detections_per_class'),
                 keep_top_k=_value_or_raise(match, pipeline_config, 'postprocessing_max_total_detections'),
                 nms_threshold=_value_or_raise(match, pipeline_config, 'postprocessing_iou_threshold')))
        # sets specific name to the node so we can find it in other replacers
        detection_output_node.name = 'detection_output'

        output_op = Output(graph, dict(name='do_OutputOp'))
        output_op.create_node([detection_output_node])

        print('The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores" '
              'have been replaced with a single layer of type "Detection Output". Refer to IR catalogue in the '
              'documentation for information about this layer.')

        return {'detection_output_node': detection_output_node}

    @staticmethod
    def do_infer(node):
        node.old_infer(node)
        # compared to the IE's DetectionOutput, the TF keeps the locations in YXYX, need to get back to the XYXY
        # for last matmul/Conv2D that operate the locations need to swap the X and Y for output feature weights & biases
        swap_weights_xy(backward_bfs_for_operation(node.in_node(0), ['MatMul', 'Conv2D']))
```

The differences in conversion are the following:

*  The locations tensor does not contain information about class 0 (background), but Inference Engine `DetectionOutput` layer expects it. Line 79 append dummy tensor with fake coordinates.
*  The prior boxes tensor are not constant like in SSDs models, so it is not possible to apply the same solution. Instead, the element-wise multiplication is added to scale prior boxes tensor values with the variances values. The attribute `variance_encoded_in_target=1` is set to the `DetectionOutput` layer (lines 141-159).
*  The X and Y coordinates in the tensor with bounding boxes locations adjustments should be swapped. For some topologies it could be done by updating preceding convolution weights, but if there is no preceding convolutional node, the Model Optimizer inserts convolution node with specific kernel and weights that performs coordinates swap during topology inference.
*  Added marker node of type `OpOutput` that is used by the Model Optimizer to determine output nodes of the topology. It is used in the dead nodes elimination pass.

#### SecondStageBoxPredictor block

The `SecondStageBoxPredictor` block differs from the self-titled block from Faster R-CNN topologies. It contains a number of `CropAndResize` operations consuming variously scaled boxes generated with a Proposal layer. The combination of `CropAndResize` layers located in the `while` loop forms a single position-sensitive ROI pooling (PSROIPooling) layer with bilinear interpolation. The `ObjectDetectionAPIPSROIPoolingReplacement` replacement matches two `while` loops with PSROIPooling layers applied to the blobs with box coordinates and classes predictions.

```python
class ObjectDetectionAPIPSROIPoolingReplacement(FrontReplacementFromConfigFileSubGraph):
    replacement_id = 'ObjectDetectionAPIPSROIPoolingReplacement'

    def run_after(self):
        return [ObjectDetectionAPIProposalReplacement]

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['output_node'].id}

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_object_detection_api_pipeline_config is None:
            raise Error(missing_param_error)
        pipeline_config = PipelineConfig(argv.tensorflow_object_detection_api_pipeline_config)
        num_classes = _value_or_raise(match, pipeline_config, 'num_classes')

        input_node = match.input_nodes(0)[0][0].in_node(0)
        if 'class_predictions' in input_node.id:
            psroipooling_output_dim = num_classes + 1
        else:
            psroipooling_output_dim = num_classes * 4

        num_spatial_bins_height = pipeline_config.get_param('num_spatial_bins_height')
        num_spatial_bins_width = pipeline_config.get_param('num_spatial_bins_width')
        crop_height = pipeline_config.get_param('crop_height')
        crop_width = pipeline_config.get_param('crop_width')
        if crop_height != crop_width:
            raise Error('Different "crop_height" and "crop_width" parameters from the pipeline config are not '
                        'supported: {} vs {}'.format(crop_height, crop_width))
        psroipooling_op = PSROIPoolingOp(graph, {'name': input_node.soft_get('name') + '/PSROIPooling',
                                                 'output_dim': psroipooling_output_dim,
                                                 'group_size': crop_width / num_spatial_bins_width,
                                                 'spatial_bins_x': num_spatial_bins_width,
                                                 'spatial_bins_y': num_spatial_bins_height,
                                                 'mode': 'bilinear',
                                                 'spatial_scale': 1,
                                                 })

        if 'reshape_swap_proposals_2d' in graph.nodes():
            reshape_swap_proposals_node = Node(graph, 'reshape_swap_proposals_2d')
        else:
            swap_proposals_node = add_convolution_to_swap_xy_coordinates(graph, Node(graph, 'proposals'), 5)
            reshape_swap_proposals_node = Reshape(graph, {'dim': [-1, 5], 'nchw_layout': True,
                                                          'name': 'reshape_swap_proposals_2d'}).create_node(
                [swap_proposals_node])
        psroipooling_node = psroipooling_op.create_node([input_node, reshape_swap_proposals_node])

        reduce_op = Reduce(graph, {'name': 'mean',
                                   'reduce_type': 'mean',
                                   'axis': int64_array([1, 2]),
                                   'keep_dims': True
                                   })
        reduce_node = reduce_op.create_node([psroipooling_node])

        graph.erase_node(match.output_node(0)[0].out_node())

        return {'output_node': reduce_node}
```

The main interest of the implementation of this replacer is the `generate_sub_graph` function.

Lines 12-15: Parses the `pipeline.config` file and gets required parameters for the `PSROIPooling` layer.
Lines 17-21: Determines number of output channels for the `PSROIPooling` layer for box coordinates and classes predictions.
Lines 23-46: Create `PSROIPooling` layer based on model parameters determined from the pipeline configuration file. 
Lines 48-53: Add Reduce layer which is the output of the `while` loops being replaced.

#### SecondStagePostprocessor block

The `SecondStagePostprocessor` block implements functionality of the `DetectionOutput` layer from the Inference Engine. The `ObjectDetectionAPIDetectionOutputReplacement` sub-graph replacement is used to replace the block. For this type of topologies the replacer adds convolution node to swap coordinates of boxes in of the 0-th input tensor to the `DetectionOutput` layer. The custom attribute `coordinates_swap_method` is set to value `add_convolution` in the sub-graph replacement configuration file to enable that behaviour. A method (`swap_weights`) is not suitable for this type of topologies because there are no `Mul` or `Conv2D` operations before the 0-th input of the `DetectionOutput` layer.

#### DetectionOutput Layer

Unlike in SSDs and Faster R-CNNs, the implementation of the `DetectionOutput` layer in Mask R-CNNs topologies is not separated in a dedicated scope. But the matcher is defined with start/end points defined in the `mask_rcnn_support.json` so the replacer correctly adds the `DetectionOutput` layer.
