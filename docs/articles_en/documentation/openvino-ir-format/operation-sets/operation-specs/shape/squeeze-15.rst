Squeeze
=======


.. meta::
  :description: Learn about Squeeze-15 - a shape manipulation operation, which
                can be performed on one required and one optional input tensor.

**Versioned name**: *Squeeze-15*

**Category**: *Shape manipulation*

**Short description**: *Squeeze* removes dimensions equal to 1 from the first input tensor.

**Detailed description**: *Squeeze* can be used with or without the second input tensor.

* If only the first input is provided, every dimension that is equal to 1 will be removed from it.
* With the second input provided, each value is an index of a dimension from the first tensor that is to be removed. Specified dimension should be equal to 1, otherwise it will be ignored and copied as is.
  Dimension indices can be specified directly, or by negative indices (counting dimensions from the end).

.. note::

    - If index of the dimension to squeeze is provided as a constant input and it points to a dynamic dimension that might be `1`, and the *allow_axis_skip* attribute is ``false``, then the dimension is considered as squeezable. Therefore the rank of the output shape will be reduced, but not dynamic. If dynamic rank is expected for such case, *allow_axis_skip* attribute need to be set to ``true``.
    - If the input with indices is empty or not provided, dynamic dimension compatible with `1` leads to dynamic rank of the output shape.


**Attributes**:

* *allow_axis_skip*

  * **Description**: If true, shape inference results in a dynamic rank if selected axis has value 1 in its dimension range.
  * **Range of values**: ``false`` or ``true``
  * **Type**: ``boolean``
  * **Required**: *no*
  * **Default value**: ``false``

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

    <layer ... type="Squeeze" version="opset15">
        <data allow_axis_skip="false"/>
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
            <port id="2" precision="FP32">
                <dim>3</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>

*Example 2: squeeze 1D tensor with 1 element to a 0D tensor (constant)*

.. code-block:: xml
   :force:

    <layer ... type="Squeeze" version="opset15">
        <data allow_axis_skip="false"/>
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
            <port id="2" precision="FP32">
            </port>
        </output>
    </layer>

*Example 3: squeeze 1D tensor with 1 dynamic shape element to a fully dynamic shape*

.. code-block:: xml
   :force:

    <layer ... type="Squeeze" version="opset15">
        <data allow_axis_skip="true"/>
        <input>
            <port id="0">
                <dim>-1</dim>
            </port>
        </input>
        <input>
            <port id="1">
                <dim>1</dim>  <!-- value is [0] -->
            </port>
        </input>
        <output>
            <port id="2" precision="FP32"/>    <!-- output with dynamic rank -->
        </output>
    </layer>

*Example 4: squeeze 2D tensor with dynamic and static shape elements to a static shape output, according to the opset1 rules*

.. code-block:: xml
   :force:

    <layer ... type="Squeeze" version="opset15">
        <data allow_axis_skip="false"/>
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>-1</dim>
            </port>
        </input>
        <input>
            <port id="1">
                <dim>1</dim>  <!-- value is [1] -->
            </port>
        </input>
        <output>
            <port id="2" precision="FP32">
                <dim>2</dim>  <!-- assumes: actual value of <dim>-1</dim> is squeezable -->
            </port>
        </output>
    </layer>

*Example 5: squeeze 2D tensor with dynamic and static shape elements to a dynamic shape output, according to the opset15 rules*

.. code-block:: xml
   :force:

    <layer ... type="Squeeze" version="opset15">
        <data allow_axis_skip="true"/>
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>-1</dim>
            </port>
        </input>
        <input>
            <port id="1">
                <dim>1</dim>  <!-- value is [1] -->
            </port>
        </input>
        <output>
            <port id="2" precision="FP32" />    <!-- Output with dynamic rank. Actual value of <dim>-1</dim> may or may not be squeezable -->
        </output>
    </layer>
