Identity
========


.. meta::
  :description: Learn about Identity-16 - a simple operation that forwards the input to the output.

**Versioned name**: *Identity-16*

**Category**: *Activation*

**Short description**: The *Identity* operation forwards the input to the output.

**Detailed description**: The *Identity* operation generates a new tensor that mirrors the input tensor in shape, data type, and content, effectively implementing the linear activation function f(x) = x.
If the input and output tensor data address is the same, input is returned as output instead.

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
                <dim>3</dim>
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="1" precision="FP32" names="Identity:16">
                <dim>3</dim>
                <dim>3</dim>
            </port>
        </output>
    </layer>
