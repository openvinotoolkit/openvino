## Erf <a name="Erf"></a> {#openvino_docs_ops_arithmetic_Erf_1}

**Versioned name**: *Erf-1*

**Category**: Arithmetic unary operation

**Short description**: *Erf* calculates the Gauss error function element-wise with given tensor.

**Detailed Description**

For each element from the input tensor calculates corresponding element in the output tensor with the following formula:
\f[
erf(x) = \pi^{-1} \int_{-x}^{x} e^{-t^2} dt
\f]

**Attributes**:

    No attributes available.

**Inputs**

* **1**: A tensor of type *T*. **Required.**

**Outputs**

* **1**: The result of element-wise operation. A tensor of type *T*.

**Types**

* *T*: any supported floating-point type.

**Examples**

*Example 1*

```xml
<layer ... type="Erf">
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
