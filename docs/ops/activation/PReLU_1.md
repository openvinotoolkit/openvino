## PReLU <a name="PReLU"></a> {#openvino_docs_ops_activation_PReLU_1}

**Versioned name**: *PReLU-1*

**Category**: *Activation function*

**Short description**: Parametric rectified linear unit element-wise activation function.

**Detailed description**

*PReLU* operation is introduced in this [article](https://arxiv.org/abs/1502.01852v1).

*PReLU* performs element-wise parametric *ReLU* operation on a given input tensor, based on the following mathematical formula:

\f[
PReLU(x) = \left\{\begin{array}{r}
    x \quad \mbox{if } x \geq  0 \\
    \alpha x \quad \mbox{if } x < 0
\end{array}\right.
\f]

where α is a learnable parameter and corresponds to the negative slope, per channel, defined by the second input `slope`.

Another mathematical representation that may be found in other references:

\f[
PReLU(x) = \max(0, x) + \alpha\cdot\min(0, x)
\f]


**Attributes**: *PReLU* operation has no attributes.

**Inputs**

* **1**: `data`. A tensor of type `T` and arbitrary shape. **Required**.
* **2**: `slope`. 1D tensor of type `T`. Tensor with negative slope values, one per channel dimension of `data` input tensor. **Required**.
* **Note**: Channels dimension corresponds to second dimension of `data` input tensor. If `data` rank is less than 2, the number of channels is 1.

**Outputs**

* **1**: The result of element-wise *PReLU* operation applied to `data` input tensor with negative slope values from `slope` input tensor. A tensor of type `T` and the same shape as `data` input tensor.

**Types**

* *T*: arbitrary supported floating-point type.

**Examples**

*Example: 1D input tensor `data`*

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

*Example: 2D input tensor `data`*

```xml
<layer ... type="Prelu">
    <input>
        <port id="0">
            <dim>20</dim>
            <dim>128</dim>
        </port>
        <port id="1">
            <dim>128</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>20</dim>
            <dim>128</dim>
        </port>
    </output>
</layer>
```

*Example: 4D input tensor `data`*

```xml
<layer ... type="Prelu">
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>20</dim>
            <dim>128</dim>
            <dim>128</dim>
        </port>
        <port id="1">
            <dim>20</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>20</dim>
            <dim>128</dim>
            <dim>128</dim>
        </port>
    </output>
</layer>
```
