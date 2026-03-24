ErfInv
======


.. meta::
  :description: Learn about ErfInv-16 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *ErfInv-16*

**Category**: *Arithmetic unary*

**Short description**: *ErfInv* performs element-wise inverse error function (erfinv) on a given input tensor.

**Detailed Description**

*ErfInv* performs element-wise inverse of the Gauss error function on a given input tensor.
For each element ``x`` in the input, the output is the value ``y`` such that ``erf(y) = x``.

.. math::

   erfinv(x) = y \text{ such that } erf(y) = \frac{2}{\sqrt{\pi}} \int_{0}^{y} e^{-t^2} dt = x

The function is defined on the open interval ``(-1, 1)`` with the following special cases:

* ``erfinv(0) = 0``
* ``erfinv(1) = +inf``
* ``erfinv(-1) = -inf``
* ``erfinv(x) = NaN`` for ``|x| > 1``

**Attributes**: *ErfInv* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *ErfInv* function applied to the input tensor. A tensor of type *T* and the same shape as the input tensor.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="ErfInv" version="opset16">
       <input>
           <port id="0">
               <dim>256</dim>
               <dim>56</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>256</dim>
               <dim>56</dim>
           </port>
       </output>
   </layer>

