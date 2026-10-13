---
sidebar_position: 2
sidebar_label: "Acos"
description: "Learn about Acos-1 - an element-wise, arithmetic operation, which can be performed on a single tensor in OpenVINO."
---

# Acos

**Versioned name**: *Acos-1*

**Category**: *Arithmetic unary*

**Short description**: *Acos* performs element-wise inverse cosine (arccos) operation with given tensor.

**Attributes**:

No attributes available.

**Inputs**

- **1**: An tensor of type *T*. **Required.**

**Outputs**

- **1**: The result of element-wise acos operation. A tensor of type *T*.

**Types**

- *T*: any numeric type.

*Acos* does the following with the input tensor *a*:

$$
a_{i} = acos(a_{i})
$$

**Examples**

*Example 1*

```xml
<layer ... type="Acos">
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
