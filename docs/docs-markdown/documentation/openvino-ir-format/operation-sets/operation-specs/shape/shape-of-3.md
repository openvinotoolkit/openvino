---
sidebar_position: 208
sidebar_label: "ShapeOf"
description: "Learn about ShapeOf-3 - a shape manipulation operation, which can be performed on an arbitrary input tensor."
---

# ShapeOf

**Versioned name**: *ShapeOf-3*

**Category**: *Shape manipulation*

**Short description**: *ShapeOf* produces 1D tensor with the input tensor shape.

**Attributes**:

- *output\_type*
  
  - **Description**: the output tensor type
  - **Range of values**: "i64" or "i32"
  - **Type**: string
  - **Default value**: "i64"
  - **Required**: *no*

**Inputs**:

- **1**: Arbitrary input tensor of type *T*. **Required.**

**Outputs**:

- **1**: 1D tensor that is equal to input tensor shape of type *T\_IND*. Number of elements is equal to input tensor rank. Can be empty 1D tensor if input tensor is a scalar, that means 0-dimensional tensor.

**Types**

- *T*: any numeric type.
- *T\_IND*: `int64` or `int32`.

**Example**

```xml
 <layer ... type="ShapeOf">
     <data output_type="i64"/>
     <input>
         <port id="0">
             <dim>2</dim>
             <dim>3</dim>
             <dim>224</dim>
             <dim>224</dim>
         </port>
     </input>
     <output>
         <port id="1">  <!-- output value is: [2,3,224,224]-->
             <dim>4</dim>
         </port>
     </output>
 </layer>
```
