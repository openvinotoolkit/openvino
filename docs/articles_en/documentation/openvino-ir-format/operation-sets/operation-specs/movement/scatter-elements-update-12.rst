ScatterElementsUpdate
=====================


**Versioned name**: *ScatterElementsUpdate-12*

**Category**: *Data movement*

**Short description**: Creates a copy of the first input tensor with elements from ``updates`` input applied according to the logic specified by *reduction* attribute and ``indices`` along the ``axis``.

**Detailed description**: Creates copy of the first input tensor, and applies the elements of ``updates`` according to the logic specified by *reduction* attribute. For each element of ``updates``, at the same index there is corresponding value of ``indices``, which is an index along dimension specified by ``axis``. The index for dimension pointed by ``axis`` is provided by values of ``indices`` input, otherwise, the index is the same as the index of the entry itself.

The dimensions of ``updates`` tensor are allowed to be less or equal to the corresponding dimensions of ``data`` tensor, but the dimension pointed by ``axis`` can be also greater (especially if the ``indices`` input contains duplicated values).

The operation to perform between the corresponding elements is specified by *reduction* attribute,
by default the elements of ``data`` tensor are simply overwritten by the values from ``updates`` input.

Additionally, *use_init_val* attribute can be used to control whether the elements from the ``data`` input tensor are used as initial value (enabled by default).

General logic of output values calculations is presented below for 1D tensor case, the element corresponding to the ``[i]`` is performed as:

.. code-block:: cpp

    output[indices[i]] = reduction(updates[i], output[indices[i]]), axis = 0

- Overwrite without additional operation, reduction = "none"

.. code-block:: cpp

    output[indices[i]] = updates[i], axis = 0

- Update by adding corresponding elements, reduction = "sum"

.. code-block:: cpp

    output[indices[i]] += updates[i], axis = 0

- Update by multiplication of the corresponding elements, reduction = "prod"

.. code-block:: cpp

    output[indices[i]] *= updates[i], axis = 0

- Update with minimum value of the corresponding elements, reduction = "min"

.. code-block:: cpp

    output[indices[i]] = min(updates[i], output[indices[i]]) axis = 0

- Update with maximum value of the corresponding elements, reduction = "max"

.. code-block:: cpp

    output[indices[i]] = max(updates[i], output[indices[i]]) axis = 0

- Update with mean value of the corresponding elements, reduction = "mean". For integer types the calculated mean is rounded down (towards negative infinity). This reduction type is not supported for the `boolean` data type.

.. code-block:: cpp

    output[indices[i]] = mean(updates[i], output[indices[i]]) axis = 0


For 2D tensor case, the update of the element corresponding to the ``[i][j]`` is performed as:

.. code-block:: cpp

    output[indices[i][j]][j] = reduction(updates[i][j], output[indices[i][j]][j]) if axis = 0
    output[i][indices[i][j]] = reduction(updates[i][j], output[indices[i][j]][j]) if axis = 1

Accordingly for 3D tensor case, the update of the element corresponding to the ``[i][j][k]`` is performed as:

.. code-block:: cpp

    output[indices[i][j][k]][j][k] = reduction(updates[i][j][k], output[indices[i][j][k]][j][k]) if axis = 0
    output[i][indices[i][j][k]][k] = reduction(updates[i][j][k], output[i][indices[i][j][k]][k]) if axis = 1
    output[i][j][indices[i][j][k]] = reduction(updates[i][j][k], output[i][j][indices[i][j][k]]) if axis = 2

**Attributes**:

* *reduction*

  * **Description**: The type of operation to perform on the inputs.
  * **Range of values**: one of ``none``, ``sum``, ``prod``, ``min``, ``max``, ``mean``
  * **Type**: `string`
  * **Default value**: ``none``
  * **Required**: *no*


* *use_init_val*

  * **Description**: Controls whether the elements in the data input tensor are used as init value for reduce operations.
  * **Range of values**:
    * true - data input elements are used
    * false - data input elements are not used
  * **Type**: boolean
  * **Default value**: true
  * **Required**: *no*
  * **Note**: The attribute has no effect for *reduction* == "none"


**Inputs**:

*   **1**: ``data`` tensor of arbitrary rank ``r`` and of type *T*. **Required.**

*   **2**: ``indices`` tensor with indices of type *T_IND*. The rank of the tensor is equal to the rank of ``data`` tensor. All index values are expected to be within bounds ``[-d, d - 1]`` along dimension ``d`` pointed by ``axis``. If multiple indices point to the same output location then the order of updating the values is undefined. Negative value of index means reverse indexing and will be normalized to value ``len(data.shape[axis] + index)``. If an index points to non-existing element then exception is raised. **Required.**

*   **3**: ``updates`` tensor of shape equal to the shape of ``indices`` tensor and of type *T*. **Required.**

*   **4**: ``axis`` tensor with scalar or 1D tensor with one element of type *T_AXIS* specifying axis for scatter. Negative ``axis`` means reverse indexing and will be normalized to value ``axis = data.rank + axis``. The value can be in range ``[-r, r - 1]`` where ``r`` is the rank of ``data``. **Required.**

**Outputs**:

*   **1**: Tensor with shape equal to ``data`` tensor of the type *T*.

**Types**

* *T*: any supported type.
* *T_IND*: any integer numeric type.
* *T_AXIS*: any integer numeric type.

* For ``boolean`` type of ``data`` input, *reduction* ``sum``, ``prod`` behaves like logical ``OR``, ``AND`` accordingly, but there is no implementation for ``boolean`` data type and *reduction* ``mean``.

**Example**

*Example 1*

.. code-block:: cpp

    <layer ... use_init_val="true" reduction="sum" type="ScatterElementsUpdate">
        <input>
            <port id="0">>  <!-- data -->
                <dim>4</dim>  <!-- values: [2, 3, 4, 6] -->
            </port>
            <port id="1">  <!-- indices (negative values allowed) -->
                <dim>6</dim>  <!-- values: [1, 0, 0, -2, -1, 2] -->
            </port>
            <port id="2">>  <!-- updates -->
                <dim>6</dim>  <!-- values: [10, 20, 30, 40, 70, 60] -->
            </port>
            <port id="3">     <!-- values: [0] -->
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="FP32">
                <dim>4</dim>  <!-- values: [52, 13, 104, 76] -->
            </port>
        </output>
    </layer>


*Example 2*

.. code-block:: cpp

    <layer ... use_init_val="false" reduction="sum" type="ScatterElementsUpdate">
        <input>
            <port id="0">>  <!-- data -->
                <dim>4</dim>  <!-- values: [2, 3, 4, 6] -->
            </port>
            <port id="1">  <!-- indices -->
                <dim>6</dim>  <!-- values: [1, 0, 0, 2, 3, 2] -->
            </port>
            <port id="2">>  <!-- updates -->
                <dim>6</dim>  <!-- values: [10, 20, 30, 40, 70, 60] -->
            </port>
            <port id="3">     <!-- values: [0] -->
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="FP32">
                <dim>4</dim>  <!-- values: [50, 10, 100, 70] -->
            </port>
        </output>
    </layer>


*Example 3*

.. code-block:: cpp

    <layer ... use_init_val="true" reduction="none" type="ScatterElementsUpdate">
        <input>
            <port id="0">>  <!-- data -->
                <dim>3</dim>
                <dim>4</dim>  <!-- values: [[0, 0, 0, 0],
                                             [0, 0, 0, 0],
                                             [0, 0, 0, 0]] -->
            </port>
            <port id="1">  <!-- indices -->
                <dim>2</dim>
                <dim>2</dim>  <!-- values: [[1, 2],
                                             [0, 3]] -->
            </port>
            <port id="2">>  <!-- updates -->
                <dim>2</dim>
                <dim>2</dim>  <!-- values: [[11, 12],
                                             [13, 14]]) -->
            </port>
            <port id="3">     <!-- values: [1] -->
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="I32">
                <dim>3</dim>
                <dim>4</dim>  <!-- values:  [[ 0, 11, 12,  0],
                                              [13,  0,  0, 14],
                                              [ 0,  0,  0,  0]] -->
            </port>
        </output>
    </layer>


*Example 4*

.. code-block:: cpp

    <layer ... use_init_val="true" reduction="sum" type="ScatterElementsUpdate">
        <input>
            <port id="0">>  <!-- data -->
                <dim>3</dim>
                <dim>4</dim>  <!-- values: [[1, 1, 1, 1],
                                             [1, 1, 1, 1],
                                             [1, 1, 1, 1]] -->
            </port>
            <port id="1">  <!-- indices -->
                <dim>2</dim>
                <dim>2</dim>  <!-- values: [[1, 1],
                                             [0, 3]] -->
            </port>
            <port id="2">>  <!-- updates -->
                <dim>2</dim>
                <dim>2</dim>  <!-- values: [[11, 12],
                                             [13, 14]]) -->
            </port>
            <port id="3">     <!-- values: [1] -->
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="I32">
                <dim>3</dim>
                <dim>4</dim>  <!-- values: [[ 1, 24,  1,  1],
                                             [14,  1,  1, 15],
                                             [ 1,  1,  1,  1]] -->
            </port>
        </output>
    </layer>


*Example 5*

.. code-block:: cpp

    <layer ... use_init_val="true" reduction="prod" type="ScatterElementsUpdate">
        <input>
            <port id="0">>  <!-- data -->
                <dim>3</dim>
                <dim>4</dim>  <!-- values: [[2, 2, 2, 2],
                                             [2, 2, 2, 2],
                                             [2, 2, 2, 2]] -->
            </port>
            <port id="1">  <!-- indices -->
                <dim>2</dim>
                <dim>2</dim>  <!-- values: [[1, 1],
                                             [0, 3]] -->
            </port>
            <port id="2">>  <!-- updates -->
                <dim>2</dim>
                <dim>2</dim>  <!-- values: [[11, 12],
                                             [13, 14]]) -->
            </port>
            <port id="3">     <!-- values: [1] -->
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="4" precision="I32">
                <dim>3</dim>
                <dim>4</dim>  <!-- values: [[  2, 264,   2,   2],
                                             [ 26,   2,   2,  28],
                                             [  2,   2,   2,   2]] -->
            </port>
        </output>
    </layer>


*Example 6*

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
            <port id="3">     <!-- values: [0] -->
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




