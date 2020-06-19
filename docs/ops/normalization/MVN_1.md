## MVN <a name="MVN"></a>

**Versioned name**: *MVN-1*

**Category**: *Normalization*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/mvn.html)

**Detailed description**

*MVN* subtracts mean value from the input blob:
\f[
o_{i} = i_{i} - \frac{\sum{i_{k}}}{C * H * W}
\f]
If *normalize_variance* is set to 1, the output blob is divided by variance:
\f[
o_{i}=\frac{o_{i}}{\sum \sqrt {o_{k}^2}+\epsilon}
\f]

**Attributes**

* *across_channels*

  * **Description**: *across_channels* is a flag that specifies whether mean values are shared across channels. For example, *across_channels* equal to `false` means that mean values are not shared across channels.
  * **Range of values**:
    * `false` - do not share mean values across channels
    * `true` - share mean values across channels
  * **Type**: `boolean`
  * **Default value**: `false`
  * **Required**: *no*

* *normalize_variance*

  * **Description**: *normalize_variance* is a flag that specifies whether to perform variance normalization.
  * **Range of values**:
    * `false` -- do not normalize variance
    * `true` -- normalize variance
  * **Type**: `boolean`
  * **Default value**: `false`
  * **Required**: *no*

* *eps*

  * **Description**: *eps* is the number to be added to the variance to avoid division by zero when normalizing the value. For example, *epsilon* equal to 0.001 means that 0.001 is added to the variance.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: 4D or 5D input tensor of any floating point type. Required.

**Outputs**

* **1**: normalized tensor of the same type and shape as input tensor.

**Example**

```xml
<layer ... type="MVN">
    <data across_channels="true" eps="1e-9" normalize_variance="true"/>
    <input>
        <port id="0">
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
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