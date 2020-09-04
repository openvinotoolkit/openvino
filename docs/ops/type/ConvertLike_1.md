## ConvertLike <a name="ConvertLike"></a> {#openvino_docs_ops_type_ConvertLike_1}

**Versioned name**: *ConvertLike-1*

**Category**: type conversion

**Short description**: Operation converts all elements of the 1st input tensor to a type of elements of 2nd input tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: `data` - A tensor of type T1. **Required.**
* **2**: `like` - A tensor of type T2. **Required.**

**Outputs**

* **1**: The result of element-wise *"ConvertLike"* operation. A tensor of the same type with `like` tensor and the same shape with `data` tensor.

**Types**

* *T1*: u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, boolean, bf16
* *T2*: u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, boolean, bf16

**Detailed description**

Conversion from one supported type to another supported type is always allowed. User must be aware of precision loss and value change caused by range difference between two types. For example, a 32-bit float *3.141592* may be round to a 32-bit int *3*.

*a* - `data` input tensor, *b* - `like` input tensor.

\f[
o_{i} = Convert[destination_type=type(b)](a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="ConvertLike">
    <input>
        <port id="0">        <!-- type: int32 -->
            <dim>256</dim>
            <dim>56</dim>
        </port>
        <port id="1">        <!-- type: float32 -->
            <dim>3</dim>     <!-- any data -->
        </port>
    </input>
    <output>
        <port id="2">        <!-- result type: float32 -->
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```