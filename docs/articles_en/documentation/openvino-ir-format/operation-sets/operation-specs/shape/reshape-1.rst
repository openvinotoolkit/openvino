Reshape
=======


.. meta::
  :description: Learn about Reshape-1 - a shape manipulation operation, which
                can be performed on two required input tensors.

**Versioned name**: *Reshape-1*

**Category**: *Shape manipulation*

**Short description**: *Reshape* operation changes dimensions of the input tensor according to the specified order. Input tensor volume is equal to output tensor volume, where volume is the product of dimensions.

**Detailed description**:

*Reshape* takes two input tensors: ``data`` to be resized and ``shape`` of the new output. The values in the ``shape`` could be ``-1``, ``0`` and any positive integer number. The two special values ``-1`` and ``0``:

* ``0`` means "copy the respective dimension *(left aligned)* of the input tensor" if ``special_zero`` is set to ``true``; otherwise it is a normal dimension and is applicable to empty tensors.
* ``-1`` means that this dimension is calculated to keep the overall elements count the same as in the input tensor. Not more than one ``-1`` can be used in a reshape operation.

If ``special_zero`` is set to ``true`` index of ``0`` cannot be larger than the rank of the input tensor.

**Attributes**:

* *special_zero*

  * **Description**: *special_zero* controls how zero values in ``shape`` are interpreted. If *special_zero* is ``false``, then ``0`` is interpreted as-is which means that output shape will contain a zero dimension at the specified location. Input and output tensors are empty in this case. If *special_zero* is ``true``, then all zeros in ``shape`` implies the copying of corresponding dimensions from ``data.shape`` into the output shape *(left aligned)*.
  * **Range of values**: ``false`` or ``true``
  * **Type**: ``boolean``
  * **Required**: *yes*

**Inputs**:

*   **1**: ``data`` a tensor of type *T* and arbitrary shape. **Required.**

*   **2**: ``shape`` 1D tensor of type *T_SHAPE* describing output shape. **Required.**

**Outputs**:

*   **1**: Output tensor of type *T* with the same content as ``data`` input tensor but with shape defined by ``shape`` input tensor.

**Types**

* *T*: any numeric type.

* *T_SHAPE*: any supported integer type.

**Examples**

*Example 1: reshape empty tensor*

.. code-block:: xml
   :force:

    <layer ... type="Reshape" ...>
        <data special_zero="false"/>
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>5</dim>
                <dim>5</dim>
                <dim>0</dim>
            </port>
            <port id="1">
                <dim>2</dim>   <!--The tensor contains 2 elements: 0, 4 -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>0</dim>
                <dim>4</dim>
            </port>
        </output>
    </layer>


*Example 2: reshape tensor - preserve first dim, calculate second and fix value for third dim*

.. code-block:: xml
   :force:

    <layer ... type="Reshape" ...>
        <data special_zero="true"/>
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>5</dim>
                <dim>5</dim>
                <dim>24</dim>
            </port>
            <port id="1">
                <dim>3</dim>   <!--The tensor contains 3 elements: 0, -1, 4 -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>2</dim>
                <dim>150</dim>
                <dim>4</dim>
            </port>
        </output>
    </layer>


*Example 3: reshape tensor - preserve first two dims, fix value for third dim and calculate fourth*

.. code-block:: xml
   :force:

    <layer ... type="Reshape" ...>
        <data special_zero="true"/>
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>2</dim>
                <dim>3</dim>
            </port>
            <port id="1">
                <dim>4</dim>   <!--The tensor contains 4 elements: 0, 0, 1, -1 -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>2</dim>
                <dim>2</dim>
                <dim>1</dim>
                <dim>3</dim>
            </port>
        </output>
    </layer>


*Example 4: reshape tensor - calculate first dim and preserve second dim*

.. code-block:: xml
   :force:

    <layer ... type="Reshape" ...>
        <data special_zero="true"/>
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>1</dim>
                <dim>1</dim>
            </port>
            <port id="1">
                <dim>2</dim>   <!--The tensor contains 2 elements: -1, 0 -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>3</dim>
                <dim>1</dim>
            </port>
        </output>
    </layer>


*Example 5: reshape tensor - preserve first dim and calculate second dim*

.. code-block:: xml
   :force:

    <layer ... type="Reshape" ...>
        <data special_zero="true"/>
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>1</dim>
                <dim>1</dim>
            </port>
            <port id="1">
                <dim>2</dim>   <!--The tensor contains 2 elements: 0, -1 -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>3</dim>
                <dim>1</dim>
            </port>
        </output>
    </layer>


