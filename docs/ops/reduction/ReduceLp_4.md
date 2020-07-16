## ReduceLp <a name="ReduceLp"></a>

**Versioned name**: *ReduceLp-4*

**Category**: *Reduction*

**Short description**: *ReduceLp* operation performs reduction with finding the Lp norm of the 1st input tensor in slices specified by the 2nd input.

**Attributes**

* *keep_dims*

  * **Description**: If set to `True` it holds axes that are used for reduction. For each such axis, output dimension is equal to 1.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: False
  * **Required**: *no*

**Inputs**

* **1**: Input tensor x of type *T1*. **Required.**

* **2**: Scalar or 1D tensor of type *T2* with axis indices for the 1st input along which reduction is performed. Accepted range is `[-r, r-1]` where where `r` is the rank of input tensor, all values must be unique, repeats are not allowed. **Required.**

* **3**: Scalar of type *T3* with value order `p` of the normalization. Possible values: `1` for L1 or `2` for L2. **Required.**

**Outputs**

* **1**: Tensor of the same type as the 1st input tensor and `shape[i] = shapeOf(input1)[i]` for all `i` that is not in the list of axes from the 2nd input. For dimensions from the 2nd input tensor, `shape[i] == 1` if `keep_dims == True`, or `i`-th dimension is removed from the output otherwise.

**Types**

* *T1*: any supported numeric type.
* *T2*: any supported integer type.
* *T3*: any supported integer type.

**Detailed Description**

Each element in the output is the result of reduction with finding a Lp norm operation along dimensions specified by the 2nd input:

   `output[i0, i1, ..., iN] = Lp[j0,..., jN](x[j0, ..., jN]))`

Where indices i0, ..., iN run through all valid indices for the 1st input and finding the Lp norm `Lp[j0, ..., jN]` have `jk = ik` for those dimensions `k` that are not in the set of indices specified by the 2nd input of the operation. 
Corner cases:

1. When the 2nd input is an empty list, then this operation does nothing, it is an identity. 
2. When the 2nd input contains all dimensions of the 1st input, this means that a single reduction scalar value is calculated for entire input tensor. 

**Example**

```xml
<layer id="1" type="ReduceLp" ...>
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
        <port id="2"/>
    </input>
    <output>
        <port id="3">
            <dim>6</dim>
            <dim>12</dim>
            <dim>1</dim>
            <dim>1</dim>
        </port>
    </output>
</layer>
```

```xml
<layer id="1" type="ReduceLp" ...>
    <data keep_dims="False" />
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
        <port id="2"/>
    </input>
    <output>
        <port id="3">
            <dim>6</dim>
            <dim>12</dim>
        </port>
    </output>
</layer>
```

```xml
<layer id="1" type="ReduceLp" ...>
    <data keep_dims="False" />
    <input>
        <port id="0">
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="1">
            <dim>1</dim>         <!-- value is [1] that means independent reduction in each channel and spatial dimensions -->
        </port>
        <port id="2"/>
    </input>
    <output>
        <port id="3">
            <dim>6</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
    </output>
</layer>
```

```xml
<layer id="1" type="ReduceLp" ...>
    <data keep_dims="False" />
    <input>
        <port id="0">
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="1">
            <dim>1</dim>         <!-- value is [-2] that means independent reduction in each channel, batch and second spatial dimension -->
        </port>
        <port id="2"/>
    </input>
    <output>
        <port id="3">
            <dim>6</dim>
            <dim>12</dim>
            <dim>24</dim>
        </port>
    </output>
</layer>
```