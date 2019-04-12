# Annotation Converters

Annotation converter is a function which converts annotation file to suitable for metric evaluation format.
Each annotation converter expects specific annotation file format or data structure, which depends on original dataset.
If converter for your data format is not supported by Accuracy Checker, you can provide your own annotation converter.
Each annotation converter has parameters available for configuration.

Process of conversion can be implemented in two ways:
* via configuration file
* via command line

### Describing annotation conversion in configuration file.

Annotation conversion can be provided in `dataset` section your configuration file to convert annotation inplace before every evaluation.
Each conversion configuration should contain `converter` field filled selected converter name and provide converter specific parameters (more details in supported converters section). All paths can be prefixed via command line with `-s, --source` argument.

You can additionally use optional parameters like:
* `subsample_size` - Dataset subsample size. You can specify the number of ground truth objects or dataset ration in percentage. Please, be careful to use this option, some datasets does not support subsampling. 
* `annotation` - path to store converted annotation pickle file. You can use this parameter if you need to reuse converted annotation to avoid subsequent conversions.
* `meta` - path to store mata information about converted annotation if it is provided.

Example of usage:
```yaml
   annotation_conversion:
     converter: sample
     data_dir: sample/sample_dataset
```


### Conversing process via command line.

The command line for annotation conversion looks like:

```bash
python3 convert_annotation.py <converter_name> <converter_specific parameters>
```
All converter specific options should have format `--<parameter_name> <parameter_value>`
You may refer to `-h, --help` to full list of command line options. Some optional arguments are:

* `-o, --output_dir` - directory to save converted annotation and meta info.
* `-a, --annotation_name` - annotation file name.
* `-m, --meta_name` - meta info file name.

### Supported converters 

Accuracy Checker supports following list of annotation converters and specific for them parameters:
* `wider` - converts from Wider Face dataset to `DetectionAnnotation`.
  * `annotation_file` - path to txt file, which contains ground truth data in WiderFace dataset format.
  * `label_start` - specifies face label index in label map. Default value is 1. You can provide another value, if you want to use this dataset for separate label validation,
  in case when your network predicts other class for faces.
* `sample` - converts annotation for SampleNet to `ClassificationAnnotation`.
  * `data_dir` - path to sample dataset root directory.
* `voc07` - converts Pascal VOC 2007 annotation for detection task to `DetectionAnnotation`.
   * `image_set_file` - path to file with validation image list.
   * `annotations_dir` - path to directory with annotation files.
   * `images_dir` - path to directory with images related to devkit root (default JPEGImages).
  * `has_background` - allows convert dataset with/without adding background_label. Accepted values are True or False. (default is True) 
* `voc_segmentation` - converts Pascal VOC annotation for semantic segmentation task to `SegmentationAnnotation`.
  * `image_set_file` - path to file with validation image list.
  * `images_dir` - path to directory with images related to devkit root (default JPEGImages).
  * `mask_dir` - path to directory with ground truth segmentation masks related to devkit root (default SegmentationClass).
* `mars` - converts MARS person reidentification dataset to `ReidentificationAnnotation`.
  * `data_dir` - path to data directory, where gallery (`bbox_test`) and `query` subdirectories are located.
* `market1501` - converts Market1501 person reidentification dataset to `ReidentificationAnnotation`.
  * `data_dir` - path to data directory, where gallery (`bounding_box_test`) and `query` subdirectories are located.
* `detection_opencv_storage` - converts detection annotation stored in Detection OpenCV storage format to `DetectionAnnotation`.
  * `annotation_file` - path to annotation in xml format.
  * `image_names_file` - path to txt file, which contains image name list for dataset.
  * `label_start` - specifies label index start in label map. Default value is 1. You can provide another value, if you want to use this dataset for separate label validation.
  * `background_label` - specifies which index will be used for background label. You can not provide this parameter if your dataset has not background label
* `face_reid_pairwise` - converts Labeled Faces in the Wild dataset for face reidentification to `ReidentificationClassificationAnnotation`.
  * `pairs_file` - path to file with annotation positive and negative pairs.
  * `train_file` - path to file with annotation positive and negative pairs used for network train (optional parameter).
  * `landmarks_file` - path to file with facial landmarks coordinates for annotation images (optional parameter).
* `landmarks_regression` - converts VGG Face 2 dataset for facial landmarks regression task to `FacialLandmarksAnnotation`.
  * `landmarks_csv_file` - path to csv file with coordinates of landmarks points.
  * `bbox_csv_file` - path to cvs file which contains bounding box coordinates for faces (optional parameter).
* `cityscapes` - converts CityScapes Dataset to `SegmentationAnnotation`.
  * `dataset_root_dir` - path to dataset root.
  * `images_subfolder` - path from dataset root to directory with validation images (Optional, default `imgsFine/leftImg8bit/val`).
  * `masks_subfolder` - path from dataset root to directory with ground truth masks (Optional, `gtFine/val`).
  * `masks_suffix` - suffix for mask file names (Optional, default `_gtFine_labelTrainIds`).
  * `images_suffix` - suffix for image file names (Optional, default `_leftImg8bit`).
  * `use_full_label_map` - allows to use full label map with 33 classes instead train label map with 18 classes (Optional, default `False`).
* `icdar15_detection` - converts ICDAR15 dataset for text detection  task to `TextDetectionAnnotation`.
  * `data_dir` - path to folder with annotations on txt format.
* `icdar13_recognition` - converts ICDAR13 dataset for text recognition task to `CharecterRecognitionAnnotation`.
  * `annotation_file` - path to annotation file in txt format.
* `mscoco_detection` - converts MS COCO dataset for object detection task to `DetectionAnnotation`.
  * `annotation_file` - path ot annotation file in json format.
  * `has_background` - allows convert dataset with/without adding background_label. Accepted values are True or False. (default is False).
  * `use_full_label_map` - allows to use original label map (with 91 object categories) from paper instead public available(80 categories).
* `mscoco_keypoints` - converts MS COCO dataset for keypoints localization task to `PoseEstimationAnnotation`.
  * `annotation_file` - path ot annotation file in json format.
* `imagenet` - convert ImageNet dataset for image classification task to `ClassificationAnnotation`.
  * `annotation_file` - path to annotation in txt format.
  * `labels_file` - path to file with word description of labels (synset words).
  * `has_background` - allows to add background label to original labels and convert dataset for 1001 classes instead 1000 (default value is False).
