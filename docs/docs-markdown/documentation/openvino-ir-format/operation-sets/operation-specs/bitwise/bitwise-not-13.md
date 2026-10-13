---
sidebar_position: 24
sidebar_label: "BitwiseNot"
description: "Learn about BitwiseNot-13 - an element-wise, bitwise negation operation, which can be performed on a single input tensor."
---

# BitwiseNot

**Versioned name**: *BitwiseNot-13*

**Category**: *Bitwise unary*

**Short description**: *BitwiseNot* performs a bitwise logical negation operation with given tensor element-wise.

**Detailed description**: *BitwiseNot* performs a bitwise logical negation operation for each element in the given tensor, based on the following algorithm.

For `boolean` type tensors, BitwiseNot is equivalent to [LogicalNot](../logical/logical-not-1.md).

If tensor is of `any supported integer` type, for each element of the tensor:

1. Convert the value from the input tensor to binary representation according to the input tensor datatype.
2. Perform a logical negation on each bit in the binary representation, where value `0` represents `false` and value `1` represents `true`.
3. Convert back the binary representation to the input datatype.

Example 1 - *BitwiseNot* output for boolean tensor:

```python
# For given boolean input:
input = [True, False]
# Perform logical negation operation same as in LogicalNot operator:
output = [False, True]
```

Example 2 - *BitwiseNot* output for uint8 tensor:

```python
# For given uint8 input:
input = [1, 3]
# Create a binary representation of uint8:
# [00000001, 00000011]
# Perform bitwise negation:
# [11111110, 11111100]
# Convert back binary values to uint8:
output = [254, 252]
```

**Attributes**: *BitwiseNot* operation has no attributes.

**Inputs**

- **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

- **1**: The result of bitwise logical negation operation. A tensor of type *T* and the same shape as the input tensor.

**Types**

- *T*: `any supported integer or boolean type`.

**Example**

```xml
<layer ... type="BitwiseNot">
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
