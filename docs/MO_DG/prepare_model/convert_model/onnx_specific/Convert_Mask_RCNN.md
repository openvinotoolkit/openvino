# Convert ONNX* Mask R-CNN Model {#openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_Mask_RCNN}

These instructions are applicable only to the Mask R-CNN model converted to the ONNX* file format from the [facebookresearch/maskrcnn-benchmark model](https://github.com/facebookresearch/maskrcnn-benchmark).

**Step 1**. Download the pre-trained model file from [onnx/models](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/mask-rcnn) (commit-SHA: 8883e49e68de7b43e263d56b9ed156dfa1e03117).

**Step 2**. To generate the Intermediate Representation (IR) of the model, change your current working directory to the Model Optimizer installation directory and run the Model Optimizer with the following parameters:
```sh
 mo \
--input_model mask_rcnn_R_50_FPN_1x.onnx \
--input "0:2" \
--input_shape [1,3,800,800] \
--mean_values [102.9801,115.9465,122.7717] \
--transformations_config front/onnx/mask_rcnn.json
```

Note that the height and width specified with the `input_shape` command line parameter could be different. Refer to the [documentation](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/mask-rcnn) for more information about supported input image dimensions and required pre- and post-processing steps.

**Step 3**. Interpret the outputs. The generated IR file has several outputs: masks, class indices, probabilities and box coordinates. The first one is a layer with the name "6849/sink_port_0". The rest three are outputs from the "DetectionOutput" layer.
