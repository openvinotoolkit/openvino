## GELU- Gaussian Error Linear Unit <a name="Gelu"></a> {#openvino_docs_ops_activation_GELU_2}

**Versioned name**: *Gelu-2*

**Category**: *Activation*

**Short description**: [Reference](https://pytorch.org/docs/stable/nn.functional.html#gelu)

**Detailed description**: [Reference](https://arxiv.org/abs/1606.08415)

**Attributes**: *Gelu* operation has no attributes.

**Mathematical Formulation**
Gelu(x)=x*Φ(x), where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.
The following equivalent combination is recognized and fused into single Gelu op: 

\f[
    Gelu(x) = 0.5*x*(1.0 + erf((x) / \sqrt{2})
\f]

Similarly, the following Gelu approximation (typical for the TensorFlow*) is recognized and fused into single Gelu op 

\f[
    Gelu(x) \approx 0.5x(1.0 + tanh(\sqrt{2.0/pi} * (x + 0.044715 * x ^ 3))
\f]

**Inputs**:

*   **1**: Multidimensional input tensor. Required.

**Example**

```xml
<layer ... type="Gelu">
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