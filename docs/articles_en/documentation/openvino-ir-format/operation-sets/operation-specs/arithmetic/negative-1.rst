Negative
========


.. meta::
  :description: Learn about  Negative-1 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Negative-1*

**Category**: *Arithmetic unary*

**Short description**: *Negative* performs element-wise negative operation on a given input tensor.

**Detailed description**

*Negative* performs element-wise negative operation on a given input tensor, based on the following mathematical formula:

.. math::

	a_{i} = -a_{i}

**Attributes**: *Negative* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *Negative* operation applied to the input tensor. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: any supported signed numeric type.

**Example**

.. code-block::  xml
   :force:

    <layer ... type="Negative">
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

