## Selu <a name="Selu"></a> {#openvino_docs_ops_arithmetic_Selu_1}

**Versioned name**: *Selu-1*

**Category**: Arithmetic unary operation

**Short description**: *Selu* calculates the SELU activation function (https://arxiv.org/abs/1706.02515) element-wise with given tensor.

**Detailed Description**

For each element from the input tensor calculates corresponding
element in the output tensor with the following formula:
\f[
selu(x) = \lambda \left\{\begin{array}{ll}
    \alpha(e^{x} - 1) \quad \mbox{if } x \le 0 \\
    x \quad \mbox{if } x >  0
\end{array}\right.
\f]

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

* **2**: `alpha` 1D tensor with one element of type T. **Required.**

* **3**: `lambda` 1D tensor with one element of type T. **Required.**

**Outputs**

* **1**: The result of element-wise operation. A tensor of type T.

**Types**

* *T*: any supported floating point type.

**Examples**

*Example 1*

```xml
<layer ... type="Selu">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
        <port id="1">
            <dim>1</dim>
        </port>
        <port id="2">
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="3">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```