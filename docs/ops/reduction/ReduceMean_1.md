## ReduceMean <a name="ReduceMean"></a> {#openvino_docs_ops_reduction_ReduceMean_1}

**Versioned name**: *ReduceMean-1*

**Category**: *Reduction*

**Short description**: *ReduceMean* operation performs reduction with finding the arithmetic mean of the 1st input tensor in slices specified by the 2nd input.

**Attributes**

* *keep_dims*

  * **Description**: If set to `true` it holds axes that are used for reduction. For each such axis, output dimension is equal to 1.
  * **Range of values**: true or false
  * **Type**: `boolean`
  * **Default value**: false
  * **Required**: *no*

**Inputs**

* **1**: Input tensor x of type *T1*. **Required.**

* **2**: Scalar or 1D tensor of type *T_IND* with axis indices for the 1st input along which reduction is performed. Accepted range is `[-r, r-1]` where where `r` is the rank of input tensor, all values must be unique, repeats are not allowed. **Required.**

**Outputs**

* **1**: Tensor of the same type as the 1st input tensor and `shape[i] = shapeOf(input1)[i]` for all `i` that is not in the list of axes from the 2nd input. For dimensions from the 2nd input tensor, `shape[i] == 1` if `keep_dims == true`, or `i`-th dimension is removed from the output otherwise.

**Types**

* *T1*: any supported numeric type.
* *T_IND*: `int64` or `int32`.

**Detailed Description**

Each element in the output is the result of arithmetic mean reduction operation along dimensions specified by the 2nd input:

    output[i0, i1, ..., iN] = mean[j0,..., jN](x[j0, ..., jN]))

Where indices i0, ..., iN run through all valid indices for the 1st input and finding the arithmetic mean `mean[j0, ..., jN]` have `jk = ik` for those dimensions `k` that are not in the set of indices specified by the 2nd input of the operation. Corner cases:

1. When the 2nd input is an empty list, then this operation does nothing, it is an identity. 
2. When the 2nd input contains all dimensions of the 1st input, this means that a single reduction value is calculated for entire input tensor. 

**Example**

```xml
<layer id="1" type="ReduceMean" ...>
    <data keep_dims="true" />
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

```xml
<layer id="1" type="ReduceMean" ...>
    <data keep_dims="false" />
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
        </port>
    </output>
</layer>
```

```xml
<layer id="1" type="ReduceMean" ...>
    <data keep_dims="false" />
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
    </input>
    <output>
        <port id="2">
            <dim>6</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
    </output>
</layer>
```

```xml
<layer id="1" type="ReduceMean" ...>
    <data keep_dims="false" />
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
    </input>
    <output>
        <port id="2">
            <dim>6</dim>
            <dim>12</dim>
            <dim>24</dim>
        </port>
    </output>
</layer>
```