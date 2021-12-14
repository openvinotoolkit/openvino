## Elu<a name="Elu"></a> {#openvino_docs_ops_activation_Elu_1}

**Versioned name**: *Elu-1*

**Category**: *Activation function*

**Short description**: Exponential linear unit element-wise activation function.

**Detailed Description**

*Elu* operation is introduced in this [article](https://arxiv.org/abs/1511.07289v3).
It performs element-wise activation function on a given input tensor, based on the following mathematical formula:

\f[
Elu(x) = \left\{\begin{array}{r}
    x \qquad \mbox{if } x >  0 \\
    \alpha(e^{x} - 1) \quad \mbox{if } x \leq 0
\end{array}\right.
\f]

where α corresponds to *alpha* attribute.

*Elu* is equivalent to *ReLU* operation when *alpha* is equal to zero.

**Attributes**

* *alpha*

  * **Description**: scale for the negative factor
  * **Range of values**: non-negative arbitrary floating-point number
  * **Type**: `float`
  * **Required**: *yes*

**Inputs**:

*   **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**:

*   **1**: The result of element-wise *Elu* function applied to the input tensor. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: arbitrary supported floating-point type.

**Example**

```xml
<layer ... type="Elu">
    <data alpha="1.0"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>128</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>128</dim>
        </port>
    </output>
</layer>
```
