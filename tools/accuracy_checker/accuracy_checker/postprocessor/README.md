# Postprocessors

Postprocessor is function which processes prediction and/or annotation data after model infer and before metric calculation. For correct work postprocessors require specific representation format. 
(e. g. clip boxes postprocessor expects detection annotation and detection prediction for processing). 

In case when you use complicated representation located in representation container, you can add options `annotation_source` and `prediction_source` in configuration file, 
if you want process only specific representations, another way postprocessor will be used for all suitable representations. `annotation_source` and `prediction_source` should contain 
comma separated list of annotation identifiers and output layer names respectively.

Every postprocessor has parameters available for configuration. 

Accuracy Checker supports following set of postprocessors:

* `cast_to_int` - casting detection bounding box coordinates given in floating point format to integer. Supported representations: `DetectionAnotation`, `DetectionPrediction`, `TextDetectionAnnotation`, `TextDetectionPrediction`.
  * `round_policy` - method for rounding: `nearest`, `greater`, `lower`, `nearest_to_zero`.
*  `clip_boxes` - clipping detection bounding box sizes. Supported representations: `DetectionAnotation`, `DetectionPrediction`.
   * `dst_width` and `dst_height` - destination width and height for box clipping respectively. You can also use `size` instead in case when destination sizes are equal.
   * `apply_to` - option which determines target boxes for processing (`annotation` for ground truth boxes and `prediction` for detection results, `all` for both).
   * `bboxes_normalized` is flag which says that target bounding boxes are in normalized format.
* `correct_yolo_v2_boxes` - resizing detection prediction bbox coordinates using specific for Yolo v2 approach. Supported representations: `DetectionAnotation`, `DetectionPrediction`.
   * `dst_width` and `dst_height` - destination width and height respectively. You can also use `size` instead in case when destination sizes are equal.
*  `encode_segmentation_mask` - encoding segmentation label image as segmentation mask. Supported representations: `SegmentationAnotation`, `SegmentationPrediction`.
*  `resize_prediction_boxes` - resizing normalized detection prediction boxes according to image size. Supported representations: `DetectionAnotation`, `DetectionPrediction`.
*  `resize_segmentation_mask` - resizing segmentation mask. Supported representations: `SegmentationAnotation`, `SegmentationPrediction`.
    * `dst_width` and `dst_height` - destination width and height for box clipping respectively. You can also use `size` instead in case when destination sizes are equal. 
       If any of these parameters are not specified, image size will be used as default.
    * `apply_to` - determines target boxes for processing (`annotation` for ground truth boxes and `prediction` for detection results, `all` for both).
*  `nms` - non-maximum suppression. Supported representations: `DetectionAnotation`, `DetectionPrediction`.
    * `overlap` - overlap threshold for merging detections.
* `filter` - filtering data using different parameters. Supported representations: `DetectionAnotation`, `DetectionPrediction`.
    * `apply_to` - determines target boxes for processing (`annotation` for ground truth boxes and `prediction` for detection results, `all` for both).
    * `remove_filtered` - removing filtered data. Annotations support ignoring filtered data without removing as default, in other cases filtered data will be removed automatically.
    * Supported parameters for filtering: `labels`, `min_confidence`, `height_range`, `width_range`, `is_empty`, `min_visibility`, `aspect_ratio`, `area_ratio`, `area_range`.
   Filtering by `height_range`, `width_range` are also available for `TextDetectionAnnotation`, `TextDetectionPrediction`, `area_range`  - for `PoseEstimationAnnotation`, `PoseEstimationPrediction` and `TextDetectionAnnotation`, `TextDetectionPrediction`.
* `normalize_landmarks_points` - normalizing ground truth landmarks points. Supported representations: `FacialLandmarksAnnotation`, `FacialLandmarksPrediction`.
    * `use_annotation_rect` - allows to use size of rectangle saved in annotation metadata for point scaling instead source image size.
* `extend_segmentation_mask` - extending annotation segmentation mask to predicted mask size making border filled by specific value. Supported representations: `SegmentationAnotation`, `SegmentationPrediction`.
  * `filling_label` - value for filling border (default 255).
* `zoom_segmentation_mask` - zooming segmentation mask. Supported representations: `SegmentationAnotation`, `SegmentationPrediction`.
  * `zoom` - size for zoom operation.
