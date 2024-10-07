GELU
====



.. meta::
  :description: Learn about Gelu-2 - an element-wise, activation operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Gelu-2*

**Category**: *Activation function*

**Short description**: Gaussian error linear unit element-wise activation function.

**Detailed description**

*Gelu* operation is introduced in this `article <https://arxiv.org/abs/1606.08415>`__.
It performs element-wise activation function on a given input tensor, based on the following mathematical formula:

.. math::

   Gelu(x) = x\cdot\Phi(x) = x\cdot\frac{1}{2}\cdot\left[1 + erf\frac{x}{\sqrt{2}}\right]

where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.

Additionally, the *Gelu* function may be approximated as follows:

.. math::

   Gelu(x) \approx 0.5\cdot x\cdot \left(1 + \tanh\left[\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)\right]\right)


**Attributes**: *Gelu* operation has no attributes.

**Inputs**:

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**:

* **1**: The result of element-wise *Gelu* function applied to the input tensor. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: arbitrary supported floating-point type.

**Example**

.. code-block:: xml
   :force:

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



