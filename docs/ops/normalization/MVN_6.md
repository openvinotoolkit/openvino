## MVN <a name="MVN"></a> {#openvino_docs_ops_normalization_MVN_6}

**Versioned name**: *MVN-6*

**Category**: *Normalization*

**Short description**: Calculates mean-variance normalization of the input tensor.

**Detailed description**

*MVN* subtracts mean value from the input blob:
\f[
o_{i} = i_{i} - \frac{\sum{i_{k}}}{C * H * W}
\f]
If *normalize_variance* is set to 1, the output blob is divided by variance:
\f[
o_{i}=\frac{o_{i}}{\sum \sqrt {o_{k}^2+\epsilon}+\epsilon}
\f]

**Attributes**

* *normalize_variance*

  * **Description**: *normalize_variance* is a flag that specifies whether to perform variance normalization.
  * **Range of values**:
    * `false` -- do not normalize variance
    * `true` -- normalize variance
  * **Type**: `boolean`
  * **Default value**: `false`
  * **Required**: *no*

* *eps*

  * **Description**: *eps* is the number to be added to the variance inside sqrt to avoid division by zero when normalizing the value. For example, *epsilon* equal to 0.001 means that 0.001 is added to the variance.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* *how_eps_applied*

  * **Description**: Choose where to add epsilon.
  * **Range of values**:
    * `in_sqrt` -- add epsilon inside sqrt
    * `outside_sqrt` -- add epsilon outside of sqrt
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: `data` - input tensor to be normalized. Type *T*. Required.

* **2**: `axes` - 1D tensor which specifies indices of dimensions in `data` that define normalization slices. Type *T_IND*. Required.

**Outputs**

* **1**: Output tensor of the same shape and type as the `data` input tensor.

**Types**

* *T*: any floating point type.

* *T_IND*: any integer type.

**Example**

```xml
<layer ... type="MVN">
    <data eps="1e-9" how_eps_applied="in_sqrt" normalize_variance="true"/>
    <input>
        <port id="0">
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="1">
            <dim>3</dim>         <!-- value of [0,2,3] means independent normalization per channels -->
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