ShapeOf
=======


.. meta::
  :description: Learn about ShapeOf-1 - a shape manipulation operation, which
                can be performed on an arbitrary input tensor.

**Versioned name**: *ShapeOf-1*

**Category**: *Shape manipulation*

**Short description**: *ShapeOf* produces 1D tensor with the input tensor shape.

**Attributes**: has no attributes.

**Inputs**:

*   **1**: Arbitrary input tensor. **Required.**

**Outputs**:

*   **1**: 1D tensor that is equal to input tensor shape. Number of elements is equal to input tensor rank. Can be empty 1D tensor if input tensor is a scalar, that mean 0-dimensional tensor.

**Example**

.. code-block:: xml
   :force:

    <layer ... type="ShapeOf">
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>3</dim>
                <dim>224</dim>
                <dim>224</dim>
            </port>
        </input>
        <output>
            <port id="1">  <!-- output value is: [2,3,224,224]-->
                <dim>4</dim>
            </port>
        </output>
    </layer>


