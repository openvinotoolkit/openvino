## Asinh <a name="Asinh"></a> {#openvino_docs_ops_arithmetic_Asinh_3}

**Versioned name**: *Asinh-3*

**Category**: Arithmetic unary operation

**Short description**: *Asinh* is inverse hyperbolic sine (arcsinh) operation.

**Detailed description**:
For each element from the input tensor calculates corresponding
element in the output tensor with the following formula:

\f[
a_{i} = asinh(a_{i})
\f]

**Attributes**: *Asinh* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise asinh operation applied to the input tensor. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: arbitrary supported floating point type.

**Examples**

*Example 1*

```xml
<layer ... type="Asinh">
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
