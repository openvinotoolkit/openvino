## ReLU <a name="ReLU"></a> {#openvino_docs_ops_activation_ReLU_1}

**Versioned name**: *ReLU-1*

**Category**: *Activation function*

**Short description**: Rectified linear unit element-wise activation function.

**Detailed description**: [Reference](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#rectified-linear-units)

*ReLU* performs element-wise activation function on a given input tensor based on the following mathematical formula:
\f[
  ReLU(x) = (x)^{+} = \max(0, x)
\f]

**Attributes**: *ReLU* operation has no attributes.

**Inputs**:

*   **1**: A tensor of type `T` and arbitrary shape. **Required**.

**Outputs**:

*   **1**: Result of element-wise *ReLU* function applied to the input tensor. A tensor of type `T` and same shape as input tensor.

**Types**

* *T*: arbitrary supported numeric type.

**Example**

```xml
<layer ... type="ReLU">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>

```
