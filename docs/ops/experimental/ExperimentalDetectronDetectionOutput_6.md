## ExperimentalDetectronDetectionOutput <a name="ExperimentalDetectronDetectionOutput"></a> {#openvino_docs_ops_experimental_ExperimentalDetectronDetectionOutput_6}

**Versioned name**: *ExperimentalDetectronDetectionOutput-6*

**Category**: Object detection

**Short description**: An operation ExperimentalDetectronDetectionOutput... TBD

**Detailed description**: TBD

**Attributes**:

* *score_threshold*

    * **Description**: *score_threshold* attribute specifies score threshold
    * **Range of values**: non-negative float
    * **Type**: float
    * **Default value**: 0.05
    * **Required**: *yes*

* *nms_threshold*

    * **Description**: *nms_threshold* attribute specifies NMS threshold
    * **Range of values**: non-negative float
    * **Type**: float
    * **Default value**: 0.5
    * **Required**: *yes*

* *num_classes*

    * **Description**: *num_classes* attribute specifies number of detected classes
    * **Range of values**: non-negative integer
    * **Type**: int
    * **Default value**: None
    * **Required**: *yes*

* *post_nms_count*

    * **Description**: *sampling_ratio* attribute specifies the maximal number of detections per class.
    * **Range of values**: non-negative integer
    * **Type**: int
    * **Default value**: 2000
    * **Required**: *yes*

* *max_detections_per_image*

    * **Description**: *max_detections_per_image* attribute specifies maximal number of detections per image
    * **Range of values**: non-negative integer
    * **Type**: int
    * **Default value**: 100
    * **Required**: *yes*

* *class_agnostic_box_regression*

    * **Description**: *class_agnostic_box_regression* attribute ia a flag specifies whether to delete background classes or not.
      * `true` means background classes should be deleted
      * `false` means background classes shouldn't be deleted.
    * **Range of values**:
    * **Type**: bool
    * **Default value**: None
    * **Required**: *yes*

* *max_delta_log_wh*

    * **Description**: *max_delta_log_wh* attribute specifies maximal delta of logarithms for width and height
    * **Range of values**:
    * **Type**: float
    * **Default value**: None
    * **Required**: *yes*

* *deltas_weights*

    * **Description**: *deltas_weights* attribute specifies deltas of weights
    * **Range of values**:
    * **Type**: float[]
    * **Default value**: None
    * **Required**: *yes*

**Inputs**

* **1**: A 2D tensor of type *T*. Input rois Input rois rank must be equal to 2. The last dimension of the 'input_rois' input must be equal to 4. The first dimension of first, second and third inputs must be the same. **Required.**

* **2**: A 2D tensor of type *T* with input deltas. The last dimension of this input must be equal to the value of the attribute `num_classes` * 4. **Required.**

* **3**: A 2D tensor of type *T* with input scores. The last dimension of this input must be equal to the value of the attribute `num_classes`. **Required.**

* **4**: A 2D tensor of type *T* with input image info. **Required.**

**Outputs**

* **1**: A 2D tensor of type *T* with shape `[max_detections_per_image, 4]` contains boxes indices.

* **2**: A 1D tensor of type *T_IND* with shape `[max_detections_per_image]` contains class indices.

* **3**: A 1D tensor of type *T* with shape `[max_detections_per_image]` contains scores indices.

* **4**: A 1D tensor of type *T_IND* with shape `[max_detections_per_image]` contains batch indices.

**Types**

* *T*: any numeric type.

* *T_IND*: `int64` or `int32`.


**Example**

```xml
<layer ... type="ExperimentalDetectronDetectionOutput">
    <data class_agnostic_box_regression="false" deltas_weights="10.0,10.0,5.0,5.0" max_delta_log_wh="4.135166645050049" max_detections_per_image="100" nms_threshold="0.5" num_classes="81" post_nms_count="2000" score_threshold="0.05000000074505806"/>
    <input>
        <port id="0">
            <dim>1000</dim>
            <dim>4</dim>
        </port>
        <port id="1">
            <dim>1000</dim>
            <dim>324</dim>
        </port>
        <port id="2">
            <dim>1000</dim>
            <dim>81</dim>
        </port>
        <port id="3">
            <dim>1</dim>
            <dim>3</dim>
        </port>
    </input>
    <output>
        <port id="4" precision="FP32">
            <dim>100</dim>
            <dim>4</dim>
        </port>
        <port id="5" precision="I32">
            <dim>100</dim>
        </port>
        <port id="6" precision="FP32">
            <dim>100</dim>
        </port>
        <port id="7" precision="I32">
            <dim>100</dim>
        </port>
    </output>
</layer>
```
