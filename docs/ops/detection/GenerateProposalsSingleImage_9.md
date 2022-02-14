# GenerateProposalsSingleImage {#openvino_docs_ops_detection_GenerateProposalsSingleImage_9}

**Versioned name**: *GenerateProposalsSingleImage-9*

**Category**: *Object detection*

**Short description**: The *GenerateProposalsSingleImage* operation computes ROIs and their scores
based on input data.

**Detailed description**: The operation performs the following steps:

1.  Transposes and reshapes predicted bounding boxes deltas and scores to get them into the same dimension order as the
anchors.
2.  Transforms anchors and deltas into proposal bboxes and clips proposal bboxes to an image. The attribute *normalized*
indicates whether the proposal bboxes are normalized or not.
3.  Sorts all `(proposal, score)` pairs by score from highest to lowest; order of pairs with equal scores is undefined.
4.  Takes top *pre_nms_count* proposals, if total number of proposals is less than *pre_nms_count* takes all proposals.
5.  Removes predicted boxes with either height or width < *min_size*.
6.  Excute nms operation and takes and returns top proposals. The number of returned proposals (output port 1 and output
port 2) is dynamic. And the max number of proposals is specified by attribute *post_nms_count*.

**Attributes**:

* *min_size*

    * **Description**: The *min_size* attribute specifies minimum box width and height.
    * **Range of values**: non-negative floating-point number
    * **Type**: float
    * **Required**: *yes*

* *nms_threshold*

    * **Description**: The *nms_threshold* attribute specifies threshold to be used in the NMS stage.
    * **Range of values**: non-negative floating-point number
    * **Type**: float
    * **Required**: *yes*

* *pre_nms_count*

    * **Description**: The *pre_nms_count* attribute specifies number of top-n proposals before NMS.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Required**: *yes*

* *post_nms_count*

    * **Description**: The *post_nms_count* attribute specifies number of top-n proposals after NMS.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Required**: *yes*

* *normalized*

    * **Description**: *normalized* is a flag that indicates whether proposal bboxes are normalized or not.
    * **Range of values**: true or false
      * *true* - the bbox coordinates are normalized.
      * *false* - the bbox coordinates are not normalized.
    * **Type**: boolean
    * **Default value**: True
    * **Required**: *no*

**Inputs**

* **1**: A 1D tensor of type *T* with 3 elements `[image_height, image_width, scale_height_and_width]` providing input
image size info. **Required.**

* **2**: A 2D tensor of type *T* with shape `[height * width * number_of_channels, 4]` providing anchors. **Required.**

* **3**: A 3D tensor of type *T* with shape `[number_of_anchors * 4, height, width]` providing deltas for anchors.
Height and width for third, fourth and fifth inputs should be equal. **Required.**

* **4**: A 3D tensor of type *T* with shape `[number_of_anchors, height, width]` providing proposals scores.  **Required.**

**Outputs**

* **1**: A 2D tensor of type *T* with dynamic shape `[N, 4]` providing ROIs. `N` equals to the actual number of proposals.

* **2**: A 1D tensor of type *T* with dynamic shape `[N]` providing ROIs scores. `N` equals to the actual number of proposals.

**Types**

* *T*: any supported floating-point type.

**Example**

```xml
<layer ... type="GenerateProposalsSingleImage" version="opset9">
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
    </input>
    <output>
        <port id="4" precision="FP32">
            <dim>1000</dim>
            <dim>4</dim>
        </port>
        <port id="5" precision="FP32">
            <dim>1000</dim>
        </port>
    </output>
</layer>
```
