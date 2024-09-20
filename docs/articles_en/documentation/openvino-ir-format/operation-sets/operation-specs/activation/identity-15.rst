Identity
=======


.. meta::
  :description: Learn about Identity-15 - a simple operation that forwards the input to the output.

**Versioned name**: *Identity-15*

**Category**: *Matrix*

**Short description**: *Identity* operation forwards the input as the output, essentially computing the4 linear activation function f(x) = x.

**Detailed description**: *Identity* operation either directly outputs the input, or returns a new Tensor with the same shape, data type and data as the input.

This behavior is targeted to mimic the design of PyTorch and Tensorflow frameworks. PyTorch by design returns the reference of the input, whereas Tensorflow creates a copy of the input.

Copy operation is significantly more memory- and computationally-heavy.

**Attribute**:

* *copy*

  * **Description**: Modifies the behavior of Identity. If false, input is passed as the output, keeping the same memory adress. If true, a copy of input is created and returned as input.
  * **Range of values**: `true`, `false`

    * ``true`` - returned value is a copy of the input. Significantly slower.
    * ``false`` - returned value is the input itself.

  * **Type**: `bool`
  * **Default value**: `false`
  * **Required**: *No*

**Input**:

* **1**: `input` - A tensor of any shape and type `T`. **Required.**

**Output**:

* **1**: `output` - A tensor with the same shape and type `T` as the input, containing the same data as the input.

**Types**

* **T**: any supported data type.

*Example 1: 2D input matrix.*

.. code-block:: xml
    :force:

    <layer ... name="Identity" type="Identity">
        <data/>
        <input>
            <port id="0" precision="FP32">
                <dim>3</dim> <!-- 3 rows of square matrix -->
                <dim>3</dim> <!-- 3 columns of square matrix -->
            </port>
        </input>
        <output>
            <port id="1" precision="FP32" names="Identity:15">
                <dim>3</dim> <!-- 3 rows of square matrix -->
                <dim>3</dim> <!-- 3 columns of square matrix -->
            </port>
        </output>
    </layer>

