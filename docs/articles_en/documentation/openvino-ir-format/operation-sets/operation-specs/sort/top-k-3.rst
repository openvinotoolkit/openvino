TopK
====


.. meta::
  :description: Learn about TopK-3 - a sorting and maximization operation,
                which can be performed on two required input tensors.

**Versioned name**: *TopK-3*

**Category**: *Sorting and maximization*

**Short description**: *TopK* computes indices and values of the *k* maximum/minimum values for each slice along specified axis.

**Attributes**

* *axis*

  * **Description**: Specifies the axis along which the values are retrieved.
  * **Range of values**: An integer. Negative value means counting dimension from the end.
  * **Type**: ``int``
  * **Required**: *yes*

* *mode*

  * **Description**: Specifies which operation is used to select the biggest element of two.
  * **Range of values**: ``min``, ``max``
  * **Type**: ``string``
  * **Required**: *yes*

* *sort*

  * **Description**: Specifies order of output elements and/or indices.
  * **Range of values**: ``value``, ``index``, ``none``
  * **Type**: ``string``
  * **Required**: *yes*

* *index_element_type*

  * **Description**: the type of output tensor with indices
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i32"
  * **Required**: *no*


**Inputs**:

*   **1**: tensor of arbitrary rank of type *T*. **Required.**

*   **2**: *k* -- scalar of any integer type specifies how many maximum/minimum elements should be computed. **Required.**

**Outputs**:

*   **1**: Output tensor of type *T* with top *k* values from the input tensor along specified dimension *axis*. The shape of the tensor is ``[input1.shape[0], ..., input1.shape[axis-1], k, input1.shape[axis+1], ...]``.

*   **2**: Output tensor with top *k* indices for each slice along *axis* dimension of type *T_IND*. The shape of the tensor is the same as for the 1st output, that is ``[input1.shape[0], ..., input1.shape[axis-1], k, input1.shape[axis+1], ...]``.

**Types**

* *T*: any numeric type.

* *T_IND*: ``int64`` or ``int32``.

**Detailed Description**

The output tensor is populated by values computed in the following way:

.. code-block:: cpp

   output[i1, ..., i(axis-1), j, i(axis+1) ..., iN] = top_k(input[i1, ...., i(axis-1), :, i(axis+1), ..., iN]), k, sort, mode)

So for each slice ``input[i1, ...., i(axis-1), :, i(axis+1), ..., iN]`` which represents 1D array, *TopK* value is computed individually.

Sorting and minimum/maximum are controlled by ``sort`` and ``mode`` attributes:

* *mode* = ``max``, *sort* = ``value`` - descending by value
* *mode* = ``max``, *sort* = ``index`` - ascending by index
* *mode* = ``max``, *sort* = ``none``  - undefined
* *mode* = ``min``, *sort* = ``value`` - ascending by value
* *mode* = ``min``, *sort* = ``index`` - ascending by index
* *mode* = ``min``, *sort* = ``none``  - undefined

If there are several elements with the same value then their output order is not determined.

**Example**

.. code-block:: cpp

  <layer ... type="TopK" ... >
      <data axis="1" mode="max" sort="value" index_element_type="i64"/>
      <input>
          <port id="0">
              <dim>6</dim>
              <dim>12</dim>
              <dim>10</dim>
              <dim>24</dim>
          </port>
          <port id="1">
              <!-- k = 3 -->
          </port>
      <output>
          <port id="2">
              <dim>6</dim>
              <dim>3</dim>
              <dim>10</dim>
              <dim>24</dim>
          </port>
          <port id="3">
              <dim>6</dim>
              <dim>3</dim>
              <dim>10</dim>
              <dim>24</dim>
          </port>
      </output>
  </layer>


