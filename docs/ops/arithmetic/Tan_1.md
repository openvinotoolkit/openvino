## Tan <a name="Tan"></a> {#openvino_docs_ops_arithmetic_Tan_1}

**Versioned name**: *Tan-1*

**Category**: Arithmetic unary operation

**Short description**: *Tan* performs element-wise tangent operation with given tensor.

**Detailed description**:  Operation takes one input tensor and performs the element-wise tangent function on a given input tensor, based on the following mathematical formula:

\f[
a_{i} = tan(a_{i})
\f]

*Example 1*

    input = [0.0, 0.25, -0.25, 0.5, -0.5]
    output = [0.0, 0.25534192, -0.25534192, 0.54630249, -0.54630249]

*Example 2*

    input = [-2, -1, 0, 1, 2]
    output = [2, -2, 0, 2, -2]

**Attributes**: *tan*  operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape, measured in radians. **Required.**

**Outputs**

* **1**: The result of element-wise *tan* applied to the input tensor. A tensor of type *T* and same shape as the input tensor.

**Types**

* *T*: int32,int64,uint32,uint64,float16,float32


**Examples**

```xml
<layer ... type="Tan">
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
