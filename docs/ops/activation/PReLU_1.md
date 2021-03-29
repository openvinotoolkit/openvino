## PReLU <a name="PReLU"></a> {#openvino_docs_ops_activation_PReLU_1}

**Versioned name**: *PReLU-1*

**Category**: *Activation function*

**Short description**: Parametric rectified linear unit element-wise activation function.

**Detailed description**

*PReLU* operation is introduced in this [article](https://arxiv.org/pdf/1502.01852v1.pdf).

*PReLU* performs element-wise parametric *ReLU* operation on a given input tensor, based on the following mathematical formula:

\f[
PReLU(x) = \left\{\begin{array}{r}
    x \quad \mbox{if } x \geq  0 \\
    \alpha x \quad \mbox{if } x < 0
\end{array}\right.
\f]

Where α, is a learnable parameter and corresponds to the negative slope defined by the second input `slope`.

Before the operation computation, the second input tensor `slope` is broadcasted to the first input tensor `data` based on [Broadcast Rules For Elementwise Operations](../broadcast_rules.md). Additionally, *PReLU* is equivalent to *ReLU* operation when `slope` is equal to zero input tensor.

**Attributes**: *PReLU* operation has no attributes.

**Inputs**

* **1**: `data`. A tensor of type `T` and arbitrary shape. **Required**.

* **2**: `slope`. A tensor of type `T` and arbitrary shape. Tensor with negative slope values. The shape of the tensor should be broadcastable to input tensor `data`. **Required**.

**Outputs**

* **1**: The result of element-wise *PReLU* operation applied to `data` input tensor with negative slope values from `slope` input tensor. A tensor of type `T` and same shape as `data` input tensor.

**Types**

* *T*: arbitrary supported floating point type.

**Example**

```xml
<layer ... type="Prelu">
    <input>
        <port id="0">
            <dim>128</dim>
        </port>
        <port id="1">
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>128</dim>
        </port>
    </output>
</layer>
```
