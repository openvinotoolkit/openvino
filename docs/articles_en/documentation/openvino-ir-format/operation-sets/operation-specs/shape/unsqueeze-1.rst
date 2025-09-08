Unsqueeze
=========


.. meta::
  :description: Learn about Unsqueeze-1 - a shape manipulation operation, which
                can be performed on two required input tensors.

**Versioned name**: *Unsqueeze-1*

**Category**: *Shape manipulation*

**Short description**: *Unsqueeze* adds dimensions of size 1 to the first input tensor. The second input value specifies a list of dimensions that will be inserted. Indices specify dimensions in the output tensor.

**Attributes**: *Unsqueeze* operation doesn't have attributes.

**Inputs**:

*   **1**: Tensor of type *T* and arbitrary shape. **Required.**

*   **2**: Scalar or 1D tensor of type *T_INT* with indices of dimensions to unsqueeze. Values could be negative (have to be from range ``[-R, R-1]``, where ``R`` is the rank of the output). **Required.**

**Outputs**:

*   **1**: Tensor with unsqueezed values of type *T*.

**Types**

* *T*: any numeric type.

* *T_INT*: any supported integer type.

**Example**

*Example 1: unsqueeze 2D tensor to a 4D tensor*

.. code-block:: xml
   :force:

    <layer ... type="Unsqueeze">
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>3</dim>
            </port>
        </input>
        <input>
            <port id="1">
                <dim>2</dim>  <!-- value is [0, 3] -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>1</dim>
                <dim>2</dim>
                <dim>3</dim>
                <dim>1</dim>
            </port>
        </output>
    </layer>


*Example 2: unsqueeze 0D tensor (constant) to 1D tensor*

.. code-block:: xml
   :force:

    <layer ... type="Unsqueeze">
        <input>
            <port id="0">
            </port>
        </input>
        <input>
            <port id="1">
                <dim>1</dim>  <!-- value is [0] -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>1</dim>
            </port>
        </output>
    </layer>


