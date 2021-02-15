## Parameter <a name="Parameter"></a> {#openvino_docs_ops_infrastructure_Parameter_1}

**Versioned name**: *Parameter-1*

**Category**: *Infrastructure*

**Short description**: *Parameter* layer specifies input to the model.

**Attributes**:

* *element_type*

  * **Description**: the type of element of output tensor
  * **Range of values**: u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, boolean, bf16
  * **Type**: string
  * **Default value**: None
  * **Required**: *Yes*

* *shape*

  * **Description**: the shape of the output tensor
  * **Range of values**: list of non-negative integers, empty list is allowed that means 0D or scalar tensor
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *Yes*

**Example**

```xml
<layer ... type="Parameter" ...>
    <data>element_type="f32" shape="1,3,224,224"</data>
    <output>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </output>
</layer>
```