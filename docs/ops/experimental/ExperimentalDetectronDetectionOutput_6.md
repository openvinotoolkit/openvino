## ExperimentalDetectronDetectionOutput <a name="ExperimentalDetectronDetectionOutput"></a> {#openvino_docs_ops_experimental_ExperimentalDetectronDetectionOutput_6}

**Versioned name**: *ExperimentalDetectronDetectionOutput-6*

**Category**:  

**Short description**: An operation ExperimentalDetectronDetectionOutput, according to the repository https://github.com/openvinotoolkit/training_extensions (see pytorch_toolkit/instance_segmentation/segmentoly/rcnn/detection_output.py).

**Detailed description**: 

**Attributes**:

* *score_threshold*

    * **Description**: *score_threshold* attribute specifies score threshold
    * **Range of values**:
    * **Type**: float
    * **Default value**:
    * **Required**: yes

* *nms_threshold*

    * **Description**: *nms_threshold* attribute specifies NMS threshold
    * **Range of values**:
    * **Type**: float
    * **Default value**:
    * **Required**: yes

* *num_classes*

    * **Description**: *num_classes* attribute specifies number of detected classes
    * **Range of values**:
    * **Type**: int
    * **Default value**:
    * **Required**: yes

* *post_nms_count*

    * **Description**: *sampling_ratio* attribute specifies the maximal number of detections per class.
    * **Range of values**:
    * **Type**: int
    * **Default value**:
    * **Required**: *yes*

* *max_detections_per_image*

    * **Description**: *max_detections_per_image* attribute specifies maximal number of detections per image
    * **Range of values**:
    * **Type**: int+
    * **Default value**:
    * **Required**: *yes*

* *class_agnostic_box_regression*

    * **Description**: *class_agnostic_box_regression* attribute ia a flag specifies whether to delete background classes or not.
      * `true` means background classes should be deleted
      * `false` means background classes shouldn't be deleted.
    * **Range of values**:
    * **Type**: bool
    * **Default value**:
    * **Required**: *yes*

* *max_delta_log_wh*

    * **Description**: *max_delta_log_wh* attribute specifies maximal delta of logarithms for width and height
    * **Range of values**:
    * **Type**: float
    * **Default value**:
    * **Required**: *yes*

* *deltas_weights*

    * **Description**: *deltas_weights* attribute specifies deltas of weights
    * **Range of values**:
    * **Type**: float[]
    * **Default value**:
    * **Required**: *yes*

**Inputs**

* **1**: A tensor of type *T*. **Required.** Input rois Input rois rank must be equal to 2. The last dimension of the 'input_rois' input must be equal to 4.

* **2**: Input deltas

* **3**: Input scores

* **4**: Input image info

**Outputs**

* **1**: The result of operation. A tensor of type *T* with . boxes indices

* **2**:  with the second output contains class indices. channels indices

* **3**:  with scores indices

* **4**:  with batch indices

**Types**

* *T*: any numeric type.

**Example**

```xml

```
