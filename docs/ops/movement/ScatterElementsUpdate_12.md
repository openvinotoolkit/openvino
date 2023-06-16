# ScatterElementsUpdate {#openvino_docs_ops_movement_ScatterElementsUpdate_12}

@sphinxdirective

**Versioned name**: *ScatterElementsUpdate-12*

**Category**: *Data movement*

**Short description**: Creates a copy of the first input tensor with elements from ``updates`` input applied according to the logic specified by *reduction* attribute and ``indices`` along the ``axis``.

**Detailed description**: Creates copy of the first input tensor, and applies the elements of ``updates`` according to the logic specified by *reduction* attribute. For each element of ``updates``, at the same index there is corresponding value of ``indices``, which is an index along dimension specified by ``axis``. The index for dimension pointed by ``axis`` is provided by values of ``indices`` input, otherwise, the index is the same as the index of the entry itself.

The dimensions of ``update`` tensor are allowed to be less or equal to the corresponding dimensions of ``data`` tensor, but the dimension pointed by ``axis`` can be also greater (especially if the ``indices`` input contains duplicated values).

The operation to perform between the corresponding elements is specified by *reduction* attribute,
by default the elements of ``data`` tensor are simply overwritten by the values from ``updates`` input.

Additionally *use_init_val* attribute can be used to control whether the elements from the ``data`` input tensor are used as initial value (enabled by default).

For instance, in a 3D tensor case, the update of the element corresponding to the ``[i][j][k]`` is performed as below:

- reduction == "copy"

.. code-block:: cpp

    output[indices[i][j][k]][j][k] = updates[i][j][k] if axis = 0,
    output[i][indices[i][j][k]][k] = updates[i][j][k] if axis = 1,
    output[i][j][indices[i][j][k]] = updates[i][j][k] if axis = 2


- reduction == "sum"

.. code-block:: cpp

    output[indices[i][j][k]][j][k] += updates[i][j][k] if axis = 0,
    output[i][indices[i][j][k]][k] += updates[i][j][k] if axis = 1,
    output[i][j][indices[i][j][k]] += updates[i][j][k] if axis = 2


- reduction == "prod"

.. code-block:: cpp

    output[indices[i][j][k]][j][k] *= updates[i][j][k] if axis = 0,
    output[i][indices[i][j][k]][k] *= updates[i][j][k] if axis = 1,
    output[i][j][indices[i][j][k]] *= updates[i][j][k] if axis = 2


- reduction == "min"

.. code-block:: cpp

    output[indices[i][j][k]][j][k] = min(updates[i][j][k], output[indices[i][j][k]][j][k]) if axis = 0,
    output[i][indices[i][j][k]][k] = min(updates[i][j][k], output[i][indices[i][j][k]][k]) if axis = 1,
    output[i][j][indices[i][j][k]] = min(updates[i][j][k], output[i][j][indices[i][j][k]]) if axis = 2


- reduction == "max"

.. code-block:: cpp

    output[indices[i][j][k]][j][k] = max(updates[i][j][k], output[indices[i][j][k]][j][k]) if axis = 0,
    output[i][indices[i][j][k]][k] = max(updates[i][j][k], output[i][indices[i][j][k]][k]) if axis = 1,
    output[i][j][indices[i][j][k]] = max(updates[i][j][k], output[i][j][indices[i][j][k]]) if axis = 2


**Attributes**:

* *reduction*

  * **Description**: The type of operation to perform on the inputs.
  * **Range of values**: one of ``copy``, ``sum``, ``prod``, ``min``, ``max``
  * **Type**: `string`
  * **Default value**: ``copy``
  * **Required**: *no*


* *use_init_val*

  * **Description**: Controls whether the elements in the data input tensor are used as init value for reduce operations.
  * **Range of values**:
    * true - data input elements are used
    * false - data input elements are not used
  * **Type**: boolean
  * **Default value**: true
  * **Required**: *no*
  * **Note**: The attribute has no effect for *reduction* == "copy"


**Inputs**:

*   **1**: ``data`` tensor of arbitrary rank ``r`` and of type *T*. **Required.**

*   **2**: ``indices`` tensor with indices of type *T_IND*. The rank of the tensor is equal to the rank of ``data`` tensor. All index values are expected to be within bounds ``[-d, d - 1]`` along dimension ``d`` pointed by ``axis``. If multiple indices point to the
same output location then the order of updating the values is undefined. If an index points to non-existing output tensor element or is negative then exception is raised. **Required.**

*   **3**: ``updates`` tensor of shape equal to the shape of ``indices`` tensor and of type *T*. **Required.**

*   **4**: ``axis`` tensor with scalar or 1D tensor with one element of type *T_AXIS* specifying axis for scatter.
The value can be in range ``[-r, r - 1]`` where ``r`` is the rank of ``data``. **Required.**

**Outputs**:

*   **1**: Tensor with shape equal to ``data`` tensor of the type *T*.

**Types**

* *T*: any numeric type.
* *T_IND*: any integer numeric type.
* *T_AXIS*: any integer numeric type.

**Example**

*Example 1*

.. code-block:: cpp

    <layer ... use_init_val="true" reduction="sum" type="ScatterElementsUpdate">
        <input>
            <port id="0">>  < !-- data -->
                <dim>4</dim>  < !-- values: [2, 3, 4, 6] -->
            </port>
            <port id="1">  < !-- indices (negative values allowed) -->
                <dim>6</dim>  < !-- values: [1, 0, 0, -2, -1, 2] -->
            </port>
            <port id="2">>  < !-- updaates -->
                <dim>6</dim>  < !-- values: [10, 20, 30, 40, 70, 60] -->
            </port>
            <port id="3">     < !-- values: [0] -->
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="FP32">
                <dim>4</dim>  < !-- values: [52, 13, 104, 76] -->
            </port>
        </output>
    </layer>


*Example 2*

.. code-block:: cpp

    <layer ... use_init_val="false" reduction="sum" type="ScatterElementsUpdate">
        <input>
            <port id="0">>  < !-- data -->
                <dim>4</dim>  < !-- values: [2, 3, 4, 6] -->
            </port>
            <port id="1">  < !-- indices -->
                <dim>6</dim>  < !-- values: [1, 0, 0, 2, 3, 2] -->
            </port>
            <port id="2">>  < !-- updaates -->
                <dim>6</dim>  < !-- values: [10, 20, 30, 40, 70, 60] -->
            </port>
            <port id="3">     < !-- values: [0] -->
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="FP32">
                <dim>4</dim>  < !-- values: [50, 10, 100, 70] -->
            </port>
        </output>
    </layer>


*Example 3*

.. code-block:: cpp

    <layer ... type="ScatterElementsUpdate">
        <input>
            <port id="0">
                <dim>1000</dim>
                <dim>256</dim>
                <dim>7</dim>
                <dim>7</dim>
            </port>
            <port id="1">
                <dim>125</dim>
                <dim>20</dim>
                <dim>7</dim>
                <dim>6</dim>
            </port>
            <port id="2">
                <dim>125</dim>
                <dim>20</dim>
                <dim>7</dim>
                <dim>6</dim>
            </port>
            <port id="3">     < !-- values: [0] -->
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="FP32">
                <dim>1000</dim>
                <dim>256</dim>
                <dim>7</dim>
                <dim>7</dim>
            </port>
        </output>
    </layer>




@endsphinxdirective
