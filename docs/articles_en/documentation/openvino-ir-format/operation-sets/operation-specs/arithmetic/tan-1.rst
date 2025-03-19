Tan
===


.. meta::
  :description: Learn about Tan-1 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Tan-1*

**Category**: *Arithmetic unary*

**Short description**: *Tan* performs element-wise tangent operation with given tensor.

**Detailed description**:  Operation takes one input tensor and performs the element-wise tangent function on a given input tensor, based on the following mathematical formula:

.. math::

   a_{i} = tan(a_{i})

*Example 1*

.. code-block:: xml
   :force:

   input = [0.0, 0.25, -0.25, 0.5, -0.5]
   output = [0.0, 0.25534192, -0.25534192, 0.54630249, -0.54630249]

*Example 2*

.. code-block:: xml
   :force:

   input = [-2, -1, 0, 1, 2]
   output = [2, -2, 0, 2, -2]

**Attributes**: *tan*  operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape, measured in radians. **Required.**

**Outputs**

* **1**: The result of element-wise *tan* applied to the input tensor. A tensor of type *T* and same shape as the input tensor.

**Types**

* *T*: any supported numeric type.


**Examples**

.. code-block:: xml
   :force:

    <layer ... type="Tan">
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

