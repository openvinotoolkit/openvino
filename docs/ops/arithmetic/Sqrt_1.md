## Sqrt <a name="Sqrt"></a> {#openvino_docs_ops_arithmetic_Sqrt_1}

**Versioned name**: *Sqrt-1*

**Category**: Arithmetic unary operation

**Short description**: Square root element-wise operation.

**Detailed description**: *Sqrt* performs element-wise square root operation on a given input tensor *a*, as in the following mathematical formula, where `o` is the output tensor:

\f[
o_{i} = \sqrt{a_{i}}
\f]


**Attributes**: *Sqrt* operation has no attributes.

**Inputs**

* **1**: A tensor of type T and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise sqrt operation. A tensor of type T and shape equal to the input shape.

**Types**

* *T*: any numeric type.

*Sqrt* does the following with the input tensor *a*:

\f[
a_{i} = \sqrt{a_{i}}
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Sqrt">
    <input>
        <port id="0">
            <dim>2</dim> <!-- input values: [4.0, 9.0] -->
        </port>
    </input>
    <output>
        <port id="1">
            <dim>2</dim> <!-- output values: [2.0, 3.0] -->
        </port>
    </output>
</layer>
```

*Example 2*

```xml
<layer ... type="Sqrt">
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
