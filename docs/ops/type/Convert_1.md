## Convert <a name="Convert"></a> {#openvino_docs_ops_type_Convert_1}

**Versioned name**: *Convert-1*

**Category**: type conversion

**Short description**: Operation converts all elements of the input tensor to a type specified in the *"destination_type"* attribute.

**Attributes**:

* *destination_type*

  * **Description**: the destination type
  * **Range of values**: one of the supported types *T*
  * **Type**: string
  * **Default value**: None
  * **Required**: *Yes*

**Inputs**

* **1**: A tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise *"Convert"* operation. A tensor of *"destination_type"* type and the same shape with input tensor.

**Types**

* *T*: u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, boolean, bf16

**Detailed description**

Conversion from one supported type to another supported type is always allowed. User must be aware of precision loss and value change caused by range difference between two types. For example, a 32-bit float *3.141592* may be round to a 32-bit int *3*.

\f[
o_{i} = convert(a_{i})
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Convert">
    <data destination_type="f32"/>
    <input>
        <port id="0">        <!-- type: i32 -->
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="1">        <!-- result type: f32 -->
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```