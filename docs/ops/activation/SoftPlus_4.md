## SoftPlus <a name="SoftPlus"></a> {#openvino_docs_ops_activation_SoftPlus_4}

**Versioned name**: *SoftPlus-4*

**Category**: *Activation function*

**Short description**: *SoftPlus* is a rectified-based element-wise activation function.

**Detailed description**

*SoftPlus* operation is introduced in this [article](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.6419).

*SoftPlus* performs element-wise activation function on a given input tensor, based on the following mathematical formula:

\f[
SoftPlus(x) = \left\{\begin{array}{r}
    x \qquad \mbox{if } x \geq threshold \\
    log(e^{x} + 1.0) \qquad \mbox{if } x < threshold
\end{array}\right.
\f]

**Note**: For numerical stability the operation reverts to the linear function when `x > threshold` where `threshold` depends on *T* and
is chosen in such a way that the difference between the linear function and exact calculation is no more than `1e-6`.
The `threshold` can be calculated with the following formula where `alpha` is the number of digits after the decimal point,
`beta` is maximum value of *T* data type:

\f[
-log(e^{10^{-\alpha}} - 1.0) < threshold < log(\beta)
\f]

For example, if *T* is `fp32`, `threshold` should be `20` or if *T* is `fp16`, `threshold` should be `11`.

**Attributes**: *SoftPlus* operation has no attributes.


**Inputs**:

*   **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**:

*   **1**: The result of element-wise *SoftPlus* function applied to the input tensor. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: arbitrary supported floating-point type.

**Example**

```xml
<layer ... type="SoftPlus">
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
