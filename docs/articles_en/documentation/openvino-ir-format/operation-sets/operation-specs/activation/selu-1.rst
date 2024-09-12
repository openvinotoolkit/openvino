Selu
====


.. meta::
  :description: Learn about SeLU-1 - an element-wise, activation operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Selu-1*

**Category**: *Activation function*

**Short description**: *Selu* is a scaled exponential linear unit element-wise activation function.

**Detailed Description**

*Selu* operation is introduced in this `article <https://arxiv.org/abs/1706.02515>`__, as activation function for self-normalizing neural networks (SNNs).

*Selu* performs element-wise activation function on a given input tensor ``data``, based on the following mathematical formula:

.. math::

   Selu(x) = \lambda \left\{\begin{array}{r} x \quad \mbox{if } x > 0 \\ \alpha(e^{x} - 1) \quad \mbox{if } x \le 0 \end{array}\right.

where α and λ correspond to inputs ``alpha`` and ``lambda`` respectively.

Another mathematical representation that may be found in other references:

.. math::

   Selu(x) = \lambda\cdot\big(\max(0, x) + \min(0, \alpha(e^{x}-1))\big)

**Attributes**: *Selu* operation has no attributes.

**Inputs**

* **1**: ``data``. A tensor of type *T* and arbitrary shape. **Required.**

* **2**: ``alpha``. 1D tensor with one element of type *T*. **Required.**

* **3**: ``lambda``. 1D tensor with one element of type *T*. **Required.**

**Outputs**

* **1**: The result of element-wise *Selu* function applied to ``data`` input tensor. A tensor of type *T* and the same shape as ``data`` input tensor.

**Types**

* *T*: arbitrary supported floating-point type.

**Example**

.. code-block::  xml
   :force:

    <layer ... type="Selu">
        <input>
            <port id="0">
                <dim>256</dim>
                <dim>56</dim>
            </port>
            <port id="1">
                <dim>1</dim>
            </port>
            <port id="2">
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="3">
                <dim>256</dim>
                <dim>56</dim>
            </port>
        </output>
    </layer>

