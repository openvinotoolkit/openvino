## ExperimentalDetectronDetectionOutput <a name="ExperimentalDetectronDetectionOutput"></a> {#openvino_docs_ops_detection_ExperimentalDetectronDetectionOutput_6}

**Versioned name**: *ExperimentalDetectronDetectionOutput-6*

**Category**: Object detection

**Short description**: An operation *ExperimentalDetectronDetectionOutput*  performs non-maximum suppression to generate the detection output using information on location and score predictions.

**Detailed description**: Apply threshold on detection probabilities and apply NMS class-wise. Leave only max_detections_per_image_ detections.

The layer has 4 inputs: tensor with input ROIs with input deltas with input scores with input data.


**Attributes**:

* *score_threshold*

    * **Description**: *score_threshold* attribute specifies threshold to consider only detections whose score are larger than a threshold.
    * **Range of values**: non-negative floating point number
    * **Type**: float
    * **Default value**: 0.05
    * **Required**: *yes*

* *nms_threshold*

    * **Description**: *nms_threshold* attribute specifies threshold to be used in the NMS stage.
    * **Range of values**: non-negative floating point number
    * **Type**: float
    * **Default value**: 0.5
    * **Required**: *yes*

* *num_classes*

    * **Description**: *num_classes* attribute specifies number of detected classes.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 81
    * **Required**: *yes*

* *post_nms_count*

    * **Description**: *sampling_ratio* attribute specifies the maximal number of detections per class.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 2000
    * **Required**: *yes*

* *max_detections_per_image*

    * **Description**: *max_detections_per_image* attribute specifies maximal number of detections per image.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 100
    * **Required**: *yes*

* *class_agnostic_box_regression*

    * **Description**: *class_agnostic_box_regression* attribute ia a flag specifies whether to delete background classes or not.
    * **Range of values**:
      * `true` means background classes should be deleted
      * `false` means background classes shouldn't be deleted
    * **Type**: boolean
    * **Default value**: false
    * **Required**: *yes*

* *max_delta_log_wh*

    * **Description**: *max_delta_log_wh* attribute specifies maximal delta of logarithms for width and height.
    * **Range of values**: floating point number
    * **Type**: float
    * **Default value**: log(1000.0f / 16.0f)
    * **Required**: *yes*

* *deltas_weights*

    * **Description**: *deltas_weights* attribute specifies deltas of weights.
    * **Range of values**: a list of non-negative floating point numbers
    * **Type**: float[]
    * **Default value**: None
    * **Required**: *yes*

**Inputs**

* **1**: A 2D tensor of type *T* with input ROIs, rank must be equal to 2. The last dimension of this input must be equal to 4. The batch dimension of first, second and third inputs must be the same. **Required.**

* **2**: A 2D tensor of type *T* with input deltas. The last dimension of this input must be equal to the value of the attribute `num_classes` * 4. **Required.**

* **3**: A 2D tensor of type *T* with input scores. The last dimension of this input must be equal to the value of the attribute `num_classes`. **Required.**

* **4**: A 2D tensor of type *T* with input data. **Required.**

**Outputs**

* **1**: A 2D tensor of type *T* with shape `[max_detections_per_image, 4]` describing boxes indices.

* **2**: A 1D tensor of type *T_IND* with shape `[max_detections_per_image]` describing class indices.

* **3**: A 1D tensor of type *T* with shape `[max_detections_per_image]` describing scores indices.

* **4**: A 1D tensor of type *T_IND* with shape `[max_detections_per_image]` describing batch indices.

**Types**

* *T*: any supported floating point type.

* *T_IND*: `int64` or `int32`.


**Example**

```xml
<layer ... type="ExperimentalDetectronDetectionOutput" version="opset6">
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
