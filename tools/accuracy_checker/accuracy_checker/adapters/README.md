# Adapters

Adapter is a function for conversion network infer output to metric specific format.
You can use 2 ways to set adapter for topology:
* Define adapter as a string.

```yml
adapter: classification
```

* Define adapter as a dictionary, using `type:` for setting adapter name. This approach gives opportunity to set additional parameters for adapter if it is required.

```yml
adapter:
  type: reid
  grn_workaround: False
```

AccuracyChecker supports following set of adapters:
* `classification` - converting output of classification model to `ClassificationPrediction` representation.
* `segmentation` - converting output of semantic segmentation model to `SeegmentationPrediction` representation.
* `tiny_yolo_v1` - converting output of Tiny YOLO v1 model to `DetectionPrediction` representation.
* `reid` - converting output of reidentification model to `ReIdentificationPrediction` representation.
  * `grn_workaround` - enabling processing output with adding Global Region Normalization layer.
* `yolo_v2` - converting output of YOLO v2 family models to `DetectionPrediction` representation.
  * `classes` - number of detection classes (default 20).
  * `anchors` - anchor values provided as comma-separated list or one of precomputed: `yolo_v2` and `tiny_yolo_v2`.
  * `coords` - number of bbox coordinates (default 4).
  * `num` - num parameter from DarkNet configuration file (default 5).
* `yolo_v3` - converting output of YOLO v3 family models to `DetectionPrediction` representation.
  * `classes` - number of detection classes (default 80).
  * `anchors` - anchor values provided as comma-separited list or precomputed: `yolo_v3`.
  * `coords` - number of bbox coordinates (default 4).
  * `num` - num parameter from DarkNet configuration file (default 3).
  * `threshold` - minimal objectness score value for valid detections (default 0.001).
  * `input_width` and `input_height` - network input width and height correspondingly (default 416).
  * `outputs` - the list of output layers names (optional), if specified there should be exactly 3 output layers provided.
* `lpr` - converting output of license plate recognition model to `CharacterRecognitionPrediction` representation.
* `ssd` - converting  output of SSD model to `DetectionPrediction` representation.
* `face_person_detection` - converting face person detection model output with 2 detection outputs to `ContainerPredition`, where value of parameters `face_out`and `person_out` are used for identification `DetectionPrediction` in container. 
  * `face_out` -  face detection output layer name.
  * `person_out` - person detection output layer name.
* `attributes_recognition`  - converting vehicle attributes recognition model output to `ContainerPrediction` where value of parameters `color_out`and `type_out` are used for identification `ClassificationPrediction` in container. 
  * `color_out` - vehicle color attribute output layer name.
  * `type_out`- vehicle type attribute output layer name.
* `head_pose` - converting head pose estimation model output to `ContainerPrediction` where names of parameters `angle_pitch`, `angle_yaw` and `angle_roll` are used for identification `RegressionPrediction` in container. 
  * `angle_pitch` - output layer name for pitch angle.
  * `angle_yaw`- output layer name for yaw angle.
  * `angle_roll` - output layer name for roll angle.
* `age_gender` - converting age gender recognition model output to `ContainerPrediction` with `ClassificationPrediction` named `gender` for gender recognition, `ClassificationPrediction` named `age_classification` and `RegressionPrediction` named `age_error` for age recognition.
  * `age_out` - output layer name for age recognition.
  * `gender_out` - output layer name for gender recognition.
* `action_detection` - converting output of model for person detection and action recognition tasks to `ContainerPrediction` with `DetectionPrdiction` for class agnostic metric calculation and `DetectionPrediction` for action recognition. The representations in container have names `class_agnostic_prediction` and `action_prediction` respectively.
  * `priorbox_out` - name of layer containing prior boxes in SSD format.
  * `loc_out` - name of layer containing box coordinates in SSD format.
  * `main_conf_out` - name of layer containing detection confidences.
  * `add_conf_out_prefix` - prefix for generation name of layers containing action confidences if topology has several following layers or layer name.
  * `add_conf_out_count` - number of layers with action confidences (optional, you can not provide this argument if action confidences contained in one layer).
  * `num_action_classes` - number classes for action recognition.
  * `detection_threshold` - minimal detection confidences level for valid detections.
* `super_resolution` - converting output of single image super resolution network to `SuperResolutionPrediction`.
* `landmarks_regression` - converting output of model for landmarks regression to `FacialLandmarksPrediction`.
* `text_detection` - converting output of model for text detection to `TextDetectionPrediction`.
  * `pixel_class_out` - name of layer containing information related to text/no-text classification for each pixel.
  * `pixel_link_out` - name of layer containing information related to linkage between pixels and their neighbors.
* `human_pose_estimation` - converting output of model for human pose estimation to `PoseEstimationPrediction`.
  * `part_affinity_fields_out` - name of output layer with keypoints pairwise relations (part affinity fields).
  * `keypoints_heatmap_out` - name of output layer with keypoints heatmaps.
* `beam_search_decoder` - realization CTC Beam Search decoder for symbol sequence recognition, converting model output to `CharacterRecognitionPrediction`.
  * `beam_size` -  size of the beam to use during decoding (default 10).
  * `blank_label` - index of the CTC blank label.
  * `softmaxed_probabilities` - indicator that model uses softmax for output layer (default False).
* `gaze_estimation` - converting output of gaze estimation model to `GazeVectorPrediction`.
