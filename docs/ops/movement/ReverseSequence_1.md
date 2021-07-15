## ReverseSequence <a name="ReverseSequence"></a> {#openvino_docs_ops_movement_ReverseSequence_1}

**Versioned name**: *ReverseSequence-1*

**Category**: data movement operation

**Short description**: *ReverseSequence* reverses variable length slices of data.

**Detailed description**: *ReverseSequence* slices input along the dimension specified in the *batch_axis*, and for each slice *i*, reverses the first *lengths[i]* (the second input) elements along the dimension specified in the *seq_axis*.

**Attributes**

* *batch_axis*

  * **Description**: *batch_axis* is the index of the batch dimension.
  * **Range of values**: an integer. Can be negative.
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* *seq_axis*

  * **Description**: *seq_axis* is the index of the sequence dimension.
  * **Range of values**: an integer. Can be negative.
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

*   **1**: tensor with input data to reverse. **Required.**

*   **2**: 1D tensor populated with integers with sequence lengths in the 1st input tensor. **Required.**

**Example**

```xml
<layer ... type="ReverseSequence">
    <data batch_axis="0" seq_axis="1"/>
    <input>
        <port id="0">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
        <port id="1">
            <dim>3</dim>
        </port>
     </input>
    <output>
        <port id="2">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
    </output>
</layer>
```
