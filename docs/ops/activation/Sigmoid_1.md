## Sigmoid<a name="Sigmoid"></a>

**Versioned name**: *Sigmoid-1*

**Category**: *Activation function*

**Short description**: Sigmoid element-wise activation function.

**Attributes**: operations has no attributes.

**Inputs**:

*   **1**: Input tensor *x* of any floating point type. Required.

**Outputs**:

*   **1**: Result of Sigmoid function applied to the input tensor *x*. Floating point tensor with shape and type matching the input tensor. Required.

**Mathematical Formulation**

   For each element from the input tensor calculates corresponding
    element in the output tensor with the following formula:
    \f[
    sigmoid( x ) = \frac{1}{1+e^{-x}}
    \f]