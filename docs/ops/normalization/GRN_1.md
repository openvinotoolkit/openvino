## GRN <a name="GRN"></a>

**Versioned name**: *GRN-1*

**Category**: *Normalization*

**Short description**: *GRN* is the Global Response Normalization with L2 norm (across channels only).

**Detailed description**:

*GRN* computes the L2 norm by channels for input tensor with shape `[N, C, ...]`. *GRN* does the following with the input tensor:

    output[i0, i1, ..., iN] = x[i0, i1, ..., iN] / sqrt(sum[j = 0..C-1](x[i0, j, ..., iN]**2) + bias)

**Attributes**:

* *bias*

  * **Description**: *bias* is added to the variance.
  * **Range of values**: a non-negative floating point value
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: Input tensor with element of any floating point type and `2 <= rank <=4`. Required.

**Outputs**

* **1**: Output tensor of the same type and shape as the input tensor.

**Example**

```xml
<layer id="5" name="normalization" type="GRN">
    <data bias="1e-4"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>20</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </input>
    <output>
        <port id="0" precision="f32">
            <dim>1</dim>
            <dim>20</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </output>
</layer>
```