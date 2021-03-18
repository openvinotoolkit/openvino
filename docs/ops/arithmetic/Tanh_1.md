## Tanh<a name="Tanh"></a> {#openvino_docs_ops_arithmetic_Tanh_1}

**Versioned name**: *Tanh-1*

**Category**: *Activation function*

**Short description**: Tanh element-wise activation function.

**Attributes**: has no attributes

**Inputs**:

*   **1**: Input tensor x of any floating point type. Required.

**Outputs**:

*   **1**: Result of Tanh function applied to the input tensor *x*. Floating point tensor with shape and type matching the input tensor.

**Detailed description**

For each element from the input tensor calculates corresponding
element in the output tensor with the following formula:
\f[
tanh ( x ) = \frac{2}{1+e^{-2x}} - 1 = 2sigmoid(2x) - 1
\f]
