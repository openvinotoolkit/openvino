## Round <a name="Round"></a> {#openvino_docs_ops_arithmetic_Round_5}

**Versioned name**: *Round-5*

**Category**: Arithmetic unary operation 

**Short description**: *Round* performs element-wise round operation with given tensor.

**Detailed description**: Operation takes one input tensor and rounds the values, element-wise, meaning it finds the nearest integer for each value. In case of halfs, the rule is to round them to the nearest even integer.
    
    round([-4.5, -1.9, -1.5, 0.9, 1.5, 2.3, 2.5, 1.5]) = [-4.0, -2.0, -2.0, 1.0, 2.0, 2.0, 2.0, 2.0]

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type *T*. **Required.**

**Outputs**

* **1**: The result of element-wise round operation. A tensor of type *T*.

**Types**

* *T*: any numeric type.

**Example**

```xml
<layer ... type="Round">
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
