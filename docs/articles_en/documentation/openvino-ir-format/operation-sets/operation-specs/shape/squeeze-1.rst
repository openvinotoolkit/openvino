Squeeze
=======


.. meta::
  :description: Learn about Squeeze-1 - a shape manipulation operation, which
                can be performed on one required and one optional input tensor.

**Versioned name**: *Squeeze-1*

**Category**: *Shape manipulation*

**Short description**: *Squeeze* removes dimensions equal to 1 from the first input tensor.

**Detailed description**: *Squeeze* can be used with or without the second input tensor.

* If only the first input is provided, every dimension that is equal to 1 will be removed from it.
* With the second input provided, each value is an index of a dimension from the first tensor that is to be removed. Specified dimension should be equal to 1, otherwise it will be ignored and copied as is.
  Dimension indices can be specified directly, or by negative indices (counting dimensions from the end).

.. note::

    Behavior before 2024.3 OpenVINO release: Error is raised when dimension to squeeze is not compatible with 1.

.. note::

    - If index of the dimension to squeeze is provided as a constant input and it points to a dynamic dimension that might be `1`, then the dimension is considered as squeezable. Therefore the rank of the output shape will be reduced, but not dynamic.
    - If the input with indices is empty or not provided, dynamic dimension compatible with `1` leads to dynamic rank of the output shape.


**Attributes**: *Squeeze* operation doesn't have attributes.

**Inputs**:

*   **1**: Multidimensional input tensor of type *T*. **Required.**

*   **2**: Scalar or 1D tensor of type *T_INT* with indices of dimensions to squeeze. Values could be negative (have to be from range ``[-R, R-1]``, where ``R`` is the rank of the first input). **Optional.**

**Outputs**:

*   **1**: Tensor with squeezed values of type *T*.

**Types**

* *T*: any numeric type.

* *T_INT*: any supported integer type.

**Example**

*Example 1: squeeze 4D tensor to a 2D tensor*

.. code-block:: xml
   :force:

    <layer ... type="Squeeze">
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>3</dim>
                <dim>1</dim>
                <dim>2</dim>
            </port>
        </input>
        <input>
            <port id="1">
                <dim>2</dim>  <!-- value [0, 2] -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>3</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>

*Example 2: squeeze 1D tensor with 1 element to a 0D tensor (constant)*

.. code-block:: xml
   :force:

    <layer ... type="Squeeze">
        <input>
            <port id="0">
                <dim>1</dim>
            </port>
        </input>
        <input>
            <port id="1">
                <dim>1</dim>  <!-- value is [0] -->
            </port>
        </input>
        <output>
            <port id="2">
            </port>
        </output>
    </layer>
