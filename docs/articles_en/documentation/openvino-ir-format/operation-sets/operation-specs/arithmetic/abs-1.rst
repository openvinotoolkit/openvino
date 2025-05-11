Abs
===


.. meta::
  :description: Learn about Abs-1 - an element-wise, arithmetic operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Abs-1*

**Category**: *Arithmetic unary*

**Short description**: *Abs* performs element-wise the absolute value with given tensor.

**Attributes**:

No attributes available.

**Inputs**

* **1**: An tensor of type *T*. **Required.**

**Outputs**

* **1**: The result of element-wise abs operation. A tensor of type *T*.

**Types**

* *T*: any numeric type.

*Abs* does the following with the input tensor *a*:

.. math::

   a_{i} = \vert a_{i} \vert


**Examples**

*Example 1*

.. code-block:: xml
   :force:

    <layer ... type="Abs">
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



