## NormalizeL2 <a name="NormalizeL2"></a>

**Versioned name**: *NormalizeL2-1*

**Category**: *Normalization*

**Short description**: *NormalizeL2* operation performs L2 normalization of the 1st input tensor in slices specified by the 2nd input.

**Attributes**

* *eps*

  * **Description**: *eps* is the number to be added/maximized to/with the variance to avoid division by zero when normalizing the value. For example, *eps* equal to 0.001 means that 0.001 is used if all the values in normalization are equal to zero.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* *eps_mode*

  * **Description**: Specifies how *eps* is combined with L2 value calculated before division.
  * **Range of values**: `add`, `max`
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: `data` - input tensor to be normalized. Type of elements is any floating point type. Required.

* **2**: `axes` - scalar or 1D tensor with axis indices for the `data` input along which L2 reduction is calculated. Required.

**Outputs**

* **1**: Tensor of the same shape and type as the `data` input and normalized slices defined by `axes` input.

**Detailed Description**

Each element in the output is the result of division of corresponding element from the `data` input tensor by the result of L2 reduction along dimensions specified by the `axes` input:

    output[i0, i1, ..., iN] = x[i0, i1, ..., iN] / sqrt(eps_mode(sum[j0,..., jN](x[j0, ..., jN]**2), eps))

Where indices `i0, ..., iN` run through all valid indices for the 1st input and summation `sum[j0, ..., jN]` have `jk = ik` for those dimensions `k` that are not in the set of indices specified by the `axes` input of the operation. One of the corner cases is when `axes` is an empty list, then we divide each input element by itself resulting value 1 for all non-zero elements. Another corner case is where `axes` input contains all dimensions from `data` tensor, which means that a single L2 reduction value is calculated for entire input tensor and each input element is divided by that value.

`eps_mode` selects how the reduction value and `eps` are combined. It can be `max` or `add` depending on `eps_mode` attribute value.

**Example**

```xml
<layer id="1" type="NormalizeL2" ...>
    <data eps="1e-8" eps_mode="add"/>
    <input>
        <port id="0">
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="1">
            <dim>2</dim>         <!-- value is [2, 3] that means independent normalization in each channel -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
    </output>
</layer>
```