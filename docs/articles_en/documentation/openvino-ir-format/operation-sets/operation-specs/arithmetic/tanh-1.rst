Tanh
====


.. meta::
  :description: Learn about Tanh-1 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Tanh-1*

**Category**: *Arithmetic unary*

**Short description**: *Tanh* performs element-wise hyperbolic tangent (tanh) operation with given tensor.

**Detailed description**

For each element from the input tensor calculates corresponding element in the output tensor with the following formula:

.. math::

   tanh ( x ) = \frac{2}{1+e^{-2x}} - 1 = 2sigmoid(2x) - 1


* For integer element type the result is rounded (half up) to the nearest integer value.

**Attributes**: *Tanh* operation has no attributes.

**Inputs**:

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**:

* **1**: The result of element-wise *Tanh* operation. A tensor of type *T* and the same shape as input tensor.
  **Types**

* *T*: any numeric type.


**Examples**

*Example 1*

.. code-block:: xml
   :force:

    <layer ... type="Tanh">
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


