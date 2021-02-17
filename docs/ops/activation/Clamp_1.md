## Clamp<a name="Clamp"></a> {#openvino_docs_ops_activation_Clamp_1}

**Versioned name**: *Clamp-1*

**Category**: *Activation function*

**Short description**: *Clamp* operation represents clipping activation function.

**Attributes**:

* *min*

  * **Description**: *min* is the lower bound of values in the output. Any value in the input that is smaller than the bound, is replaced with the *min* value. For example, *min* equal 10 means that any value in the input that is smaller than the bound, is replaced by 10.
  * **Range of values**: non-negative positive floating point number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

* *max*

  * **Description**: *max* is the upper bound of values in the output. Any value in the input that is greater than the bound, is replaced with the *max* value. For example, *max* equals 50 means that any value in the input that is greater than the bound, is replaced by 50.
  * **Range of values**: positive floating point number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: Multidimensional input tensor. Required.

**Outputs**:

*   **1**: Multidimensional output tensor with shape and type matching the input tensor.

**Detailed description**:

*Clamp* does the following with the input tensor element-wise:
\f[
clamp( x )=\left\{\begin{array}{ll}
    max\_value \quad \mbox{if } \quad input( x )>max\_value \\
    min\_value \quad \mbox{if } \quad input( x )
\end{array}\right.
\f]

**Example**

```xml
<layer ... type="Clamp" ... >
    <data min="10" max="50" />
    <input> ... </input>
    <output> ... </output>
</layer>
```
