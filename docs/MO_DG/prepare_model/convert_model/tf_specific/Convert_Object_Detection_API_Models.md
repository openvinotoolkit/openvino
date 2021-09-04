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

#### SecondStagePostprocessor block

The `SecondStagePostprocessor` block implements functionality of the `DetectionOutput` layer from the Inference Engine. The `ObjectDetectionAPIDetectionOutputReplacement` sub-graph replacement is used to replace the block. For this type of topologies the replacer adds convolution node to swap coordinates of boxes in of the 0-th input tensor to the `DetectionOutput` layer. The custom attribute `coordinates_swap_method` is set to value `add_convolution` in the sub-graph replacement configuration file to enable that behaviour. A method (`swap_weights`) is not suitable for this type of topologies because there are no `Mul` or `Conv2D` operations before the 0-th input of the `DetectionOutput` layer.

#### DetectionOutput Layer

Unlike in SSDs and Faster R-CNNs, the implementation of the `DetectionOutput` layer in Mask R-CNNs topologies is not separated in a dedicated scope. But the matcher is defined with start/end points defined in the `mask_rcnn_support.json` so the replacer correctly adds the `DetectionOutput` layer.
