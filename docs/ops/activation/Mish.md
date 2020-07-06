## Mish <a name="Mish"></a>

**Versioned name**: *Mish-4*

**Category**: *Activation*

**Short description**: Mish a Self Regularized Non-Monotonic Neural Activation Function.

**Detailed description**: Mish layer computes self regularized non-monotonic neural activation function: x * tanh(softplus(x))

**Attributes**: operation has no attributes.

**Inputs**:

*   **1**: Input tensor *x* of any floating point type T. Required.

**Outputs**:

*   **1**: Result of computes Mish activation. Floating point tensor with shape and type matching the input tensor. Required.

**Types**

* *T*: any floating point type.

**Mathematical Formulation**

   For each element from the input tensor calculates corresponding
    element in the output tensor with the following formula:
    \f[
    Mish( x ) = x*tanh{log{1+e^{x}}}
    \f]

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