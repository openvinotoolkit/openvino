## ReduceLogicalAnd <a name="ReduceLogicalAnd"></a>

**Versioned name**: *ReduceLogicalAnd-1*

**Category**: *Reduction*

**Short description**: *ReduceLogicalAnd* operation performs reduction with *logical and* operation of the 1st input tensor in slices specified by the 2nd input.

**Attributes**

* *keep_dims*

  * **Description**: If set to `True` it holds axes that are used for reduction. For each such axis, output dimension is equal to 1.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: False
  * **Required**: *no*

**Inputs**

* **1**: Input tensor x of any data type that has defined *logical and* operation. **Required.**

* **2**: Scalar or 1D tensor with axis indices for the 1st input along which reduction is performed. **Required.**

**Outputs**

* **1**: Tensor of the same type as the 1st input tensor and `shape[i] = shapeOf(input1)[i]` for all `i` that is not in the list of axes from the 2nd input. For dimensions from the 2nd input tensor, `shape[i] == 1` if `keep_dims == True`, or `i`-th dimension is removed from the output otherwise.

**Detailed Description**

Each element in the output is the result of reduction with *logical and* operation along dimensions specified by the 2nd input:

    output[i0, i1, ..., iN] = and[j0,..., jN](x[j0, ..., jN]**2))

Where indices i0, ..., iN run through all valid indices for the 1st input and *logical and* operation `and[j0, ..., jN]` have `jk = ik` for those dimensions `k` that are not in the set of indices specified by the 2nd input of the operation. 
Corner cases:
    1. When the 2nd input is an empty list, then this operation does nothing, it is an identity. 
    2. When the 2nd input contains all dimensions of the 1st input, this means that a single reduction value is calculated for entire input tensor. 

**Example**

```xml
<layer id="1" type="ReduceLogicalAnd" ...>
    <data keep_dims="True" />
    <input>
        <port id="0">
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="1">
            <dim>2</dim>         <!-- value is [2, 3] that means independent reduction in each channel and batch -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>6</dim>
            <dim>12</dim>
            <dim>1</dim>
            <dim>1</dim>
        </port>
    </output>
</layer>
```