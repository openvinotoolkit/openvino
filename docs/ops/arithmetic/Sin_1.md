## Sin <a name="Sin"></a> {#openvino_docs_ops_arithmetic_Sin_1}

**Versioned name**: *Sin-1*

**Category**: Arithmetic unary operation 

**Short description**: *Sin* performs element-wise sine operation with given tensor.

**Detailed description**: *sin* does the following with the input tensor *a*:
\f[
a_{i} = sin(a_{i})
\f]

The sine of an acute angle theta is defined in the context of a right triangle: for the specified angle, it is the ratio of the length of the side 
that is opposite that angle (the opposite), to the length of the longest side of the triangle (the hypotenuse).

\f[
sin_{\theta} = \frac{Opposite}{Hypotenuse}
\f]

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise sin operation. A tensor of type T.

**Types**

* *T*: any numeric type.


**Examples**

*Example 1*

```xml
<layer ... type="Sin">
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
