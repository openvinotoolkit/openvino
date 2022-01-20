# ExperimentalDetectronGenerateProposalsSingleImage {#openvino_docs_ops_detection_ExperimentalDetectronGenerateProposalsSingleImage_8}

**Versioned name**: *ExperimentalDetectronGenerateProposalsSingleImage-8*

**Category**: *Object detection*

**Short description**: The *ExperimentalDetectronGenerateProposalsSingleImage* operation computes ROIs and their scores
based on input data.

**Detailed description**: The operation performs the following steps:

1.  Transposes and reshapes predicted bounding boxes deltas, variances and scores to get them into the same order as the
anchors.
2.  Transforms anchors into proposals using deltas, variances and clips proposals to an image.
3.  Removes predicted boxes with either height or width < *min_size*.
4.  Sorts all `(proposal, score)` pairs by score from highest to lowest; order of pairs with equal scores is undefined.
5.  Takes top *pre_nms_count* proposals, if total number of proposals is less than *pre_nms_count* takes all proposals.
6.  Applies non-maximum suppression with *adaptive_nms_threshold*. The initial value of *adaptive_nms_threshold* is
*nms_threshold*. If `nms_eta < 1` and `adaptive_threshold > 0.5`, update `adaptive_threshold *= nms_eta`.
7.  Takes top *post_nms_count* proposals and returns these top proposals, scores and the number of proposals. If
attribute *dynamic_output* is false and total number of proposals is less than *post_nms_count* returns output tensors
filled with zeroes. If *dynamic_output* is false, then need not to fill output tensors with zeros. No matter attribute
*dynamic_output* is true or false, the value will be the actual number of proposals.

**Attributes**:

* *min_size*

    * **Description**: The *min_size* attribute specifies minimum box width and height.
    * **Range of values**: non-negative floating-point number
    * **Type**: float
    * **Default value**: None
    * **Required**: *yes*

* *nms_threshold*

    * **Description**: The *nms_threshold* attribute specifies threshold to be used in the NMS stage.
    * **Range of values**: non-negative floating-point number
    * **Type**: float
    * **Default value**: None
    * **Required**: *yes*

* *pre_nms_count*

    * **Description**: The *pre_nms_count* attribute specifies number of top-n proposals before NMS.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: None
    * **Required**: *yes*

* *post_nms_count*

    * **Description**: The *post_nms_count* attribute specifies number of top-n proposals after NMS.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: None
    * **Required**: *yes*

* *coordinates_offset*

    * **Description**: The *coordinates_offset* attribute specifies the relationship between ROI location points and ROI width and height. For example if *coordinates_offset* is false, the `width = x_right - x_left`. If *coordinates_offset* is true, the `width = x_right - x_left + 1`.
    * **Range of values**: true or false
    * **Type**: boolean
    * **Default value**: false
    * **Required**: *no*

* *nms_eta*

    * **Description**: eta parameter for adaptive NMS.
    * **Range of values**: a floating-point number in close range `[0, 1.0]`.
    * **Type**: float
    * **Default value**: `1.0`
    * **Required**: *no*

* *dynamic_output*

    * **Description**: The *dynamic_output* attribute specifies whether the output ROIs and ROIs scores are dynamic.
    * **Range of values**: true or false
    * **Type**: boolean
    * **Default value**: false
    * **Required**: *no*


**Inputs**

* **1**: A 1D tensor of type *T* with 3 elements `[image_height, image_width, scale_height_and_width]` providing input
image size info. **Required.**

* **2**: A 2D tensor of type *T* with shape `[height * width * number_of_channels, 4]` providing anchors. **Required.**

* **3**: A 3D tensor of type *T* with shape `[number_of_channels * 4, height, width]` providing deltas for anchors.
Height and width for third, fourth and fifth inputs should be equal. **Required.**

* **4**: A 3D tensor of type *T* with shape `[number_of_channels, height, width]` providing proposals scores.
**Required.**

* **5**: A 3D tensor of type *T* with shape `[number_of_channels * 4, height, width]` providing variances for anchors.
Height and width for third, fourth and fifth inputs should be equal. Default values should be 1.0. **Optional.**

**Outputs**

* **1**: A 2D tensor of type *T* with shape `[N, 4]` providing ROIs. `N` equals to *post_nms_count* if *dynamic_output*
is false. `N` equals to the actual number of proposals if *dynamic_output* is true.

* **2**: A 1D tensor of type *T* with shape `[N]` providing ROIs scores. `N` equals to *post_nms_count* if *dynamic_output*
is false. `N` equals to the actual number of proposals if *dynamic_output* is true.

* **3**: A 1D tensor of type *int64* with shape `[1]` providing the actual number of proposals.

**Types**

* *T*: any supported floating-point type.

**Example**

```xml
<layer ... type="ExperimentalDetectronGenerateProposalsSingleImage" version="opset8">
    <data min_size="0.0" nms_threshold="0.699999988079071" post_nms_count="1000" pre_nms_count="1000"/>
    <input>
        <port id="0">
            <dim>3</dim>
        </port>
        <port id="1">
            <dim>12600</dim>
            <dim>4</dim>
        </port>
        <port id="2">
            <dim>12</dim>
            <dim>50</dim>
            <dim>84</dim>
        </port>
        <port id="3">
            <dim>3</dim>
            <dim>50</dim>
            <dim>84</dim>
        </port>
        <port id="4">
            <dim>12</dim>
            <dim>50</dim>
            <dim>84</dim>
        </port>
    </input>
    <output>
        <port id="5" precision="FP32">
            <dim>1000</dim>
            <dim>4</dim>
        </port>
        <port id="6" precision="FP32">
            <dim>1000</dim>
        </port>
        <port id="7" precision="I64">
            <dim>1</dim>
        </port>
    </output>
</layer>
```
