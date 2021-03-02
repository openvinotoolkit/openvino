## GELU- Gaussian Error Linear Unit <a name="Gelu"></a> {#openvino_docs_ops_activation_GELU_7}

**Versioned name**: *Gelu-7*

**Category**: *Activation*

**Short description**: Calculates Gaussian error linear.

**Detailed description**: `Gelu(x) = x * Φ(x)`, where `Φ(x)` is the Cumulative Distribution Function for Gaussian Distribution.
The Gelu operation is introduced in the [paper](https://arxiv.org/abs/1606.08415).

**Attributes**

* *approximation_mode*

  * **Description**: Specifies the formulae to calculate the output.
  * **Range of values**:
    * `erf` -- calculate output using the Gauss error function.
    * `tanh` -- calculate output using tanh approximation
  * **Type**: `string`
  * **Default value**: `erf`
  * **Required**: *no*


**Mathematical Formulation**

For the `erf` approximation mode:
\f[
    Gelu(x) = 0.5 \cdot x \cdot (1.0 + erf((x) / \sqrt{2})
\f]

For the `tanh` approximation mode:

\f[
    Gelu(x) \approx 0.5 \cdot x \cdot (1.0 + tanh(\sqrt{2.0/pi} \cdot (x + 0.044715 \cdot x ^ 3))
\f]

**Inputs**:

*   **1**: Multidimensional input tensor of type *T*. Required.

**Outputs**:

*   **1**: Floating point tensor with shape and type *T* matching the input tensor.

**Types**

* *T*: any floating point type.

**Examples**

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
