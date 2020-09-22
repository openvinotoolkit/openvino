## LogSoftMax <a name="LogSoftmax"></a> {#openvino_docs_ops_activation_LogSoftmax_5}

**Versioned name**: *LogSoftmax-5*

**Category**: *Activation*

**Short description**: LogSoftmax computes the log of softmax values for the given input.

**Attributes**

* *axis*

  * **Description**: *axis* represents the axis of which the *LogSoftmax* is calculated. *axis* equal 1 is a default value.
  * **Range of values**: positive integer value
  * **Type**: int
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

*   **1**: Input tensor with enough number of dimension to be compatible with *axis* attribute. Required.

**Outputs**:

*   **1**: The resulting tensor of the same shape and type as input tensor.

**Mathematical Formulation**

\f[
y_{c} = log\left(\frac{e^{Z_{c}}}{\sum_{d=1}^{C}e^{Z_{d}}}\right)
\f]
where \f$C\f$ is a size of tensor along *axis* dimension.

**Example**

```xml
<layer ... type="LogSoftmax" ... >
    <data axis="1" />
    <input> ... </input>
    <output> ... </output>
</layer>
```