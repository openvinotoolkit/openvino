## ExperimentalDetectronGenerateProposalsSingleImage <a name="ExperimentalDetectronGenerateProposalsSingleImage"></a> {#openvino_docs_ops_detection_ExperimentalDetectronGenerateProposalsSingleImage_6}

**Versioned name**: *ExperimentalDetectronGenerateProposalsSingleImage-6*

**Category**: Object detection

**Short description**: An operation *ExperimentalDetectronGenerateProposalsSingleImage* computes ROIs and their scores based on input data.

**Detailed description**: Operation doing next steps:

1.  Transposes and reshape predicted deltas and scores to get them into the same order as the anchors;
2.  Transforms anchors into proposals and clips proposals to image;
3.  Removes predicted boxes with either height or width < *min_size*;
4.  Sorts all `(proposal, score)` pairs by score from highest to lowest;
5.  Takes top *pre_nms_count* proposals;
6.  Applies non maximum suppression with *nms_threshold*;
7.  Takes *post_nms_count* proposals and return these top proposals and their scores.
       
**Attributes**:

* *min_size*

    * **Description**: *min_size* attribute specifies minimum box width & height.
    * **Range of values**: non-negative floating point number
    * **Type**: float
    * **Default value**: 0.0
    * **Required**: *yes*

* *nms_threshold*

    * **Description**: *nms_threshold* attribute specifies threshold to be used in the NMS stage.
    * **Range of values**: non-negative floating point number
    * **Type**: float
    * **Default value**: 0.7
    * **Required**: *yes*

* *post_nms_count*

    * **Description**: *post_nms_count* attribute specifies number of top-n proposals after NMS.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 1000
    * **Required**: *yes*

* *pre_nms_count*

    * **Description**: *pre_nms_count* attribute specifies number of top-n proposals before NMS.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 1000
    * **Required**: *yes*

**Inputs**

* **1**: A 1D tensor of type *T* with shape `[3]` with input data. **Required.**

* **2**: A 2D tensor of type *T* with input anchors. The second dimension of this input should be 4. **Required.**

* **3**: A 3D tensor of type *T* with input deltas. Height and width for third and fourth inputs must be equal. **Required.**

* **4**: A 3D tensor of type *T* with input scores. **Required.**

**Outputs**

* **1**: A 2D tensor of type *T* with shape `[post_nms_count, 4]` describing ROIs.

* **2**: A 1D tensor of type *T* with shape `[post_nms_count]` describing ROIs scores.

**Types**

* *T*: any supported floating point type.

**Example**

```xml
<layer ... type="ExperimentalDetectronGenerateProposalsSingleImage" version="opset6">
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
