---
sidebar_position: 120
sidebar_label: "LogicalNot"
description: "Learn about LogicalNot-1 - an element-wise, logical binary operation, which can be performed on a single input tensor."
---

# LogicalNot

**Versioned name**: *LogicalNot-1*

**Category**: *Logical unary*

**Short description**: *LogicalNot* performs element-wise logical negation operation with given tensor.

**Detailed description**: *LogicalNot* performs element-wise logical negation operation with given tensor, based on the following mathematical formula:

$$
a_{i} = \lnot a_{i}
$$

**Attributes**: *LogicalNot* operation has no attributes.

**Inputs**

- **1**: A tensor of type *T\_BOOL* and arbitrary shape. **Required.**

**Outputs**

- **1**: The result of element-wise logical negation operation. A tensor of type *T\_BOOL* and the same shape as input tensor.

**Types**

- *T\_BOOL*: `boolean`.

$$
a_{i} = \lnot a_{i}
$$

**Example**

```xml
<layer ... type="LogicalNot">
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
