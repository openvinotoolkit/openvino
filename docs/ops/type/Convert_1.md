## Convert <a name="Convert"></a> {#openvino_docs_ops_type_Convert_1}

**Versioned name**: *Convert-1*

**Category**: *Type conversion*

**Short description**: *Convert* operation performs element-wise conversion on a given input tensor to a type specified in the *destination_type* attribute.

**Detailed description**

Conversion from one supported type to another supported type is always allowed. User must be aware of precision loss and value change caused by range difference between two types. For example, a 32-bit float `3.141592` may be round to a 32-bit int `3`. The result of unsupported conversions is undefined, e.g. conversion of negative signed integer value to any unsigned integer type.

Output elements are represented as follows:

\f[
o_{i} = Convert(a_{i})
\f]

where `a` corresponds to the input tensor.

**Attributes**:

* *destination_type*

  * **Description**: the destination type.
  * **Range of values**: one of the supported types *T*
  * **Type**: `string`
  * **Required**: *yes*

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *Convert* operation. A tensor of *destination_type* type and the same shape as input tensor.

**Types**

* *T*: any supported type

**Example**

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
