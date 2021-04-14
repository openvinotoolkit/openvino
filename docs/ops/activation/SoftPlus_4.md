## SoftPlus <a name="SoftPlus"></a> {#openvino_docs_ops_activation_SoftPlus_4}

**Versioned name**: *SoftPlus-4*

**Category**: *Activation*

**Short description**: SoftPlus takes one input tensor and produces output tensor where the softplus function is applied to the tensor elementwise.

**Detailed description**: For each element from the input tensor calculates corresponding
element in the output tensor with the following formula:

\f[
SoftPlus(x) = \left\{\begin{array}{r}
    x \qquad \mbox{if } x \geq threshold \\
    log(e^{x} + 1.0) \qquad \mbox{if } x < threshold
\end{array}\right.
\f]

**Note**: For numerical stability the operation reverts the linear function when `x > threshold` where `threshold` depends on *T* and
is chosen in such a way that the difference between the linear function and exact calculation is no more than `1e-6`.
The `threshold` can be calculated with the following formula where `alpha` is the count of the numbers after the decimal point:

\f[
threshold > -log(e^{10^{\alpha}} - 1.0)
\f]

For example, if *T* is `fp32`, `threshold` should be `20` or if *T* is `fp16`, `threshold` should be `12`.

**Attributes**: *SoftPlus* operation has no attributes.


**Inputs**:

*   **1**: Multidimensional input tensor of type *T*. **Required**.

**Outputs**:

*   **1**: The resulting tensor of the same shape and type as input tensor.

**Types**

* *T*: arbitrary supported floating point type.


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