## Clamp<a name="Clamp"></a> {#openvino_docs_ops_activation_Clamp_1}

**Versioned name**: *Clamp-1*

**Category**: *Activation function*

**Short description**: *Clamp* operation represents clipping activation function.

**Detailed description**:

*Clamp* performs clipping operation over the input tensor element-wise. Element values of the output are within the range `[min, max]`.
* Input values that are smaller than *min* are replaced with *min* value. For example, *min* equal 10 means that any value in the input that is smaller than the bound, is replaced by 10.
* Input values that are greater than *max* are replaced with the *max* value. For example, *max* equals 50 means that any value in the input that is greater than the bound, is replaced by 50.
* Input values within the range `[min, max]` remain unchanged.

Mathematical formula of *Clamp* is as follows:
\f[
clamp( x )=\left\{\begin{array}{ll}
    input( x ) \quad \mbox{if } \quad min \leq input( x ) \leq max \\
    max \quad \mbox{if } \quad input( x ) > max \\
    min \quad \mbox{if } \quad input( x ) < min
\end{array}\right.
\f]

**Attributes**:

* *min*

  * **Description**: *min* is the lower bound of values in the output.
  * **Range of values**: positive floating point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* *max*

  * **Description**: *max* is the upper bound of values in the output.
  * **Range of values**: positive floating point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: A tensor of type `T` and arbitrary shape. **Required**.

**Outputs**:

*   **1**: A tensor of type `T` with same shape as input tensor.

**Types**

* *T*: any numeric type.

**Example**

```xml
<layer id="1" name="clamp_node" type="Clamp">
    <data min="10" max="50" />
    <input>
        <port id="0">
            <dim>256</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>256</dim>
        </port>
    </output>
</layer>
```
