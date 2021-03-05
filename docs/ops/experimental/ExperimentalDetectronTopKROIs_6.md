## ExperimentalDetectronTopKROIs <a name="ExperimentalDetectronTopKROIs"></a> {#openvino_docs_ops_experimental_ExperimentalDetectronTopKROIs_6}

**Versioned name**: *ExperimentalDetectronTopKROIs-6*

**Category**: Sort

**Short description**: An operation ExperimentalDetectronTopKROIs is TopK operation applied to input ROIs

**Detailed description**: TBD

**Attributes**:

* *max_rois*

    * **Description**: *max_rois* attribute specifies maximal numbers of output ROIs
    * **Range of values**: non-negative integer
    * **Type**: `int`
    * **Default value**: 0
    * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T1* with input rois. **Required.**

* **2**: A tensor of type *T2* with probabilities for input ROIs. **Required.**

**Outputs**

* **1**: The result of operation. A tensor of type *T1* with shape `[max_rois, 4]`.

**Types**

* *T1*: any numeric type.

* *T2*: any floating point type.

**Example**

```xml
<layer ... type="ExperimentalDetectronTopKROIs">
    <data max_rois="1000"/>
    <input>
        <port id="0">
            <dim>5000</dim>
            <dim>4</dim>
        </port>
        <port id="1">
            <dim>5000</dim>
        </port>
    </input>
    <output>
        <port id="2" precision="FP32">
            <dim>1000</dim>
            <dim>4</dim>
        </port>
    </output>
</layer>
```
