# Floor  {#openvino_docs_ops_arithmetic_Floor_1}

**Versioned name**: *Floor-1*

**Category**: *Arithmetic unary*

**Short description**: *Floor* performs element-wise floor operation with given tensor.

**Detailed description**: For each element from the input tensor calculates corresponding
element in the output tensor with the following formula:

\f[
a_{i} = \lfloor a_{i} \rfloor
\f]

**Attributes**: *Floor* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise floor operation. A tensor of type *T*.

**Types**

* *T*: any numeric type.


**Examples**

*Example 1*

```xml
<layer ... type="Floor">
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
