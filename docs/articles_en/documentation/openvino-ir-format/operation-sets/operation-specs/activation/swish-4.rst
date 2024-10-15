Swish
=====


.. meta::
  :description: Learn about Swish-4 - an element-wise, activation operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Swish-4*

**Category**: *Activation function*

**Short description**: *Swish* performs element-wise activation function on a given input tensor.

**Detailed description**

*Swish* operation is introduced in this `article <https://arxiv.org/abs/1710.05941>`__.

*Swish* is a smooth, non-monotonic function. The non-monotonicity property of *Swish* distinguishes itself from most common activation functions. It performs element-wise activation function on a given input tensor,  based on the following mathematical formula:

.. math::

   Swish(x) = x\cdot \sigma(\beta x) = x \left(1 + e^{-(\beta x)}\right)^{-1}

where β corresponds to ``beta`` scalar input.

**Attributes**: *Swish* operation has no attributes.

**Inputs**:

*   **1**: ``data``. A tensor of type *T* and arbitrary shape. **Required.**

*   **2**: ``beta``. A non-negative scalar value of type *T*. Multiplication parameter for the sigmoid. Default value 1.0 is used. **Optional.**

**Outputs**:

*   **1**: The result of element-wise *Swish* function applied to the input tensor ``data``. A tensor of type *T* and the same shape as ``data`` input tensor.

**Types**

* *T*: arbitrary supported floating-point type.

**Examples**

Example: Second input ``beta`` provided

.. code-block:: xml
   :force:

    <layer ... type="Swish">
        <input>
            <port id="0">
                <dim>256</dim>
                <dim>56</dim>
            </port>
            <port id="1">  <!-- beta value: 2.0 -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>256</dim>
                <dim>56</dim>
            </port>
        </output>
    </layer>


Example: Second input ``beta`` not provided

.. code-block:: xml
   :force:

    <layer ... type="Swish">
        <input>
            <port id="0">
                <dim>128</dim>
            </port>
        </input>
        <output>
            <port id="1">
                <dim>128</dim>
            </port>
        </output>
    </layer>

