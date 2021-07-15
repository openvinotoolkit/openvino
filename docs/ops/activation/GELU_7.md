## GELU- Gaussian Error Linear Unit <a name="Gelu"></a> {#openvino_docs_ops_activation_GELU_7}

**Versioned name**: *Gelu-7*

**Category**: *Activation function*

**Short description**: Gaussian error linear unit element-wise activation function.

**Detailed description**:

*Gelu* operation is introduced in this [article](https://arxiv.org/abs/1606.08415).
It performs element-wise activation function on a given input tensor, based on the following mathematical formula:

\f[
    Gelu(x) = x\cdot\Phi(x)
\f]

where `Φ(x)` is the Cumulative Distribution Function for Gaussian Distribution.

The *Gelu* function may be approximated in two different ways based on *approximation_mode* attribute.

For `erf` approximation mode, *Gelu* function is represented as:

\f[
    Gelu(x) = x\cdot\Phi(x) = x\cdot\frac{1}{2}\cdot\left[1 + erf\left(x/\sqrt{2}\right)\right]
\f]

For `tanh` approximation mode, *Gelu* function is represented as:

\f[
    Gelu(x) \approx x\cdot\frac{1}{2}\cdot \left(1 + \tanh\left[\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)\right]\right)
\f]

**Attributes**

* *approximation_mode*

  * **Description**: Specifies the formulae to calculate the *Gelu* function.
  * **Range of values**:
    * `erf` - calculate output using the Gauss error function
    * `tanh` - calculate output using tanh approximation
  * **Type**: `string`
  * **Default value**: `erf`
  * **Required**: *no*

**Inputs**:

*   **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**:

*   **1**: The result of element-wise *Gelu* function applied to the input tensor. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: arbitrary supported floating-point type.

**Examples**

*Example: `tanh` approximation mode*

```xml
<layer ... type="Gelu">
    <data approximation_mode="tanh"/>
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

*Example: `erf` approximation mode*

```xml
<layer ... type="Gelu">
    <data approximation_mode="erf"/>
    <input>
        <port id="0">
            <dim>3</dim>
            <dim>7</dim>
            <dim>9</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>3</dim>
            <dim>7</dim>
            <dim>9</dim>
        </port>
    </output>
</layer>

```
