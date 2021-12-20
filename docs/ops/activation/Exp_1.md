## Exp<a name="Exp"></a> {#openvino_docs_ops_activation_Exp_1}

**Versioned name**: *Exp-1*

**Category**: *Activation function*

**Short description**: Exponential element-wise activation function.

**Detailed description**

*Exp* performs element-wise exponential activation function on a given input tensor. The mathematical formula is as follows:

\f[
exp(x) = e^{x}
\f]

**Attributes**: *Exp* operation has no attributes.

**Inputs**

*   **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

*   **1**: The result of element-wise *Exp* function applied to the input tensor. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: arbitrary supported floating-point type.

**Example**

```xml
<layer ... type="Exp">
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>256</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>256</dim>
        </port>
    </output>
</layer>
```
