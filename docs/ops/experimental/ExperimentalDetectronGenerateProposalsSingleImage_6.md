## ExperimentalDetectronGenerateProposalsSingleImage <a name="ExperimentalDetectronGenerateProposalsSingleImage"></a> {#openvino_docs_ops_experimental_ExperimentalDetectronGenerateProposalsSingleImage_6}

**Versioned name**: *ExperimentalDetectronGenerateProposalsSingleImage*

**Category**: Generation

**Short description**: An operation ExperimentalDetectronGenerateProposalsSingleImage... TBD

**Detailed description**: TBD

**Attributes**:

* *min_size*

    * **Description**: *min_size* attribute specifies minimum box width & height
    * **Range of values**: non-negative float
    * **Type**: float
    * **Default value**: 0.0
    * **Required**: *yes*

* *nms_threshold*

    * **Description**: *nms_threshold* attribute specifies NMS threshold
    * **Range of values**: non-negative float
    * **Type**: float
    * **Default value**: 0.7
    * **Required**: *yes*

* *post_nms_count*

    * **Description**: *post_nms_count* attribute specifies number of top-n proposals after NMS
    * **Range of values**: non-negative integer
    * **Type**: int
    * **Default value**: 1000
    * **Required**: *yes*

* *pre_nms_count*

    * **Description**: *pre_nms_count* attribute specifies number of top-n proposals before NMS
    * **Range of values**: non-negative integer
    * **Type**: int
    * **Default value**: 1000
    * **Required**: *yes*

**Inputs**

* **1**: A 1D tensor of type *T* with shape `[3]` with image info. **Required.**

* **2**: A 2D tensor of type *T* with input anchors. The second dimension of 'input_anchors' should be 4. **Required.**

* **3**: A 3D tensor of type *T* with input deltas. Height and width for third and fourth inputs must be equal. **Required.** 

* **4**: A 3D tensor of type *T* with input scores. **Required.**

**Outputs**

* **1**: A 2D tensor of type *T* with shape `[post_nms_count, 4]`.

* **2**: A 1D tensor of type *T* with shape `[post_nms_count]`.

**Types**

* *T*: any numeric type.

**Example**

```xml
<layer ... type="ExperimentalDetectronGenerateProposalsSingleImage">
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
