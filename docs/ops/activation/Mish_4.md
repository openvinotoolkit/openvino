## Mish <a name="Mish"></a> {#openvino_docs_ops_activation_Mish_4}

**Versioned name**: *Mish-4*

**Category**: *Activation function*

**Short description**: *Mish* is a Self Regularized Non-Monotonic Neural Activation Function.

**Detailed description**

*Mish* is a self regularized non-monotonic neural activation function proposed in this [article](https://arxiv.org/abs/1908.08681).

*Mish* performs element-wise activation function on a given input tensor, based on the following mathematical formula:

\f[
Mish(x) = x\cdot\tanh\big(Softplus(x)\big) = x\cdot\tanh\big(\ln(1+e^{x})\big)
\f]

**Attributes**: *Mish* operation has no attributes.

**Inputs**:

*   **1**: A tensor of type `T` and arbitrary shape. **Required**.

**Outputs**:

*   **1**: The result of element-wise *Mish* function applied to the input tensor. A tensor of type `T` and same shape as input tensor.

**Types**

* *T*: arbitrary supported floating point type.

**Examples**

```xml
<layer ... type="Mish">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="3">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```
