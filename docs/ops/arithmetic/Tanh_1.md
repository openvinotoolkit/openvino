## Tanh<a name="Tanh"></a> {#openvino_docs_ops_arithmetic_Tanh_1}

**Versioned name**: *Tanh-1*

**Category**: *Activation function*

**Short description**: Tanh unary arithmetic function.

**Attributes**:

    No attributes available.

**Inputs**:

* **1**: A tensor of type T. **Required.**

**Outputs**:

* **1**: The result of element-wise *Tanh* operation. A tensor of type *T* and the same shape as input tensor.
**Types**

* *T*: any numeric type.

**Detailed description**

For each element from the input tensor calculates corresponding
element in the output tensor with the following formula:
\f[
tanh ( x ) = \frac{2}{1+e^{-2x}} - 1 = 2sigmoid(2x) - 1
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Tanh">
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
