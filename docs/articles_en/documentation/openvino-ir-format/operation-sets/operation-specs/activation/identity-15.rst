Identity
========


.. meta::
  :description: Learn about Identity-15 - a simple operation that forwards the input to the output.

**Versioned name**: *Identity-15*

**Category**: *Matrix*

**Short description**: The *Identity* operation forwards the input to the output.

**Detailed description**: The *Identity* operation generates a new tensor that mirrors the input tensor in shape, data type, and content, effectively implementing the linear activation function f(x) = x.
This operation creates a copy of the input data, therefore any modifications to the output tensor do not impact the original input tensor.

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
