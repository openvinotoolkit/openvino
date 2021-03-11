## ExperimentalDetectronTopKROIs <a name="ExperimentalDetectronTopKROIs"></a> {#openvino_docs_ops_experimental_ExperimentalDetectronTopKROIs_6}

**Versioned name**: *ExperimentalDetectronTopKROIs-6*

**Category**: Sort

**Short description**: An operation *ExperimentalDetectronTopKROIs* is TopK operation applied to probabilities of input ROIs

**Detailed description**: Operation performs probabilities sorting for input ROIs and returns *max_rois* number of ROIs.

**Attributes**:

* *max_rois*

    * **Description**: *max_rois* attribute specifies maximal numbers of output ROIs.
    * **Range of values**: non-negative integer number
    * **Type**: int
    * **Default value**: 0
    * **Required**: *yes*

**Inputs**

* **1**: A 2D tensor of type *T* with shape `[number_of_input_ROIs, 4]` contains input ROIs. **Required.**

* **2**: A 1D tensor of type *T* with probabilities for input ROIs. Number of ROIs and number of probabilities should be equal. **Required.**

**Outputs**

* **1**: A 2D tensor of type *T* with shape `[max_rois, 4]` describing *max_rois* ROIs with top probabilities.

**Types**

* *T*: any supported floating point type.

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
