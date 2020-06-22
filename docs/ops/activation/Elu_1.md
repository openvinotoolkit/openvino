## Elu<a name="Elu"></a>

**Versioned name**: *Elu-1*

**Category**: *Activation function*

**Short description**: Exponential linear unit element-wise activation function.

**Detailed Description**

For each element from the input tensor calculates corresponding
element in the output tensor with the following formula:
\f[
elu(x) = \left\{\begin{array}{ll}
    alpha(e^{x} - 1) \quad \mbox{if } x < 0 \\
    x \quad \mbox{if } x \geq  0
\end{array}\right.
\f]

**Attributes**

* *alpha*

  * **Description**: scale for the negative factor
  * **Range of values**: arbitrary floating point number
  * **Type**: float
  * **Default value**: none
  * **Required**: *yes*

**Inputs**:

*   **1**: Input tensor x of any floating point type. Required.

**Outputs**:

*   **1**: Result of Elu function applied to the input tensor *x*. Floating point tensor with shape and type matching the input tensor. Required.
