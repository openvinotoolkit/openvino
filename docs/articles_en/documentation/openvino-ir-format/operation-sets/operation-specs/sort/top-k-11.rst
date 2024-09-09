TopK
====


.. meta::
  :description: Learn about TopK-11 - a sorting and maximization operation,
                which can be performed on two required input tensors.

**Versioned name**: *TopK-11*

**Category**: *sorting and maximization*

**Short description**: *TopK* computes indices and values of the *k* maximum/minimum values for each slice along a specified axis.

**Attributes**

* *axis*

  * **Description**: Specifies the axis along which the values are retrieved.
  * **Range of values**: An integer. Negative values means counting dimension from the back.
  * **Type**: ``int``
  * **Required**: *yes*

* *mode*

  * **Description**: Specifies whether *TopK* selects the largest or the smallest elements from each slice.
  * **Range of values**: "min", "max"
  * **Type**: ``string``
  * **Required**: *yes*

* *sort*

  * **Description**: Specifies the order of corresponding elements of the output tensor.
  * **Range of values**: ``value``, ``index``, ``none``
  * **Type**: ``string``
  * **Required**: *yes*

* *stable*

  * **Description**: Specifies whether the equivalent elements should maintain their relative order from the input tensor. Takes effect only if the ``sort`` attribute is set to ``value`` or ``index``.
  * **Range of values**: *true* of *false*
  * **Type**: ``boolean``
  * **Default value**: *false*
  * **Required**: *no*

* *index_element_type*

  * **Description**: the type of output tensor with indices
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i32"
  * **Required**: *no*


**Inputs**:

*   **1**: tensor with arbitrary rank and type *T*. **Required.**

*   **2**: The value of *K* - a scalar of any integer type that specifies how many elements from the input tensor should be selected. The accepted range of values of *K* is ``<1;input1.shape[axis]>``. The behavior of this operator is undefined if the value of *K* does not meet those requirements. **Required.**

**Outputs**:

*   **1**: Output tensor of type *T* with *k* values from the input tensor along a specified *axis*. The shape of the tensor is ``[input1.shape[0], ..., input1.shape[axis-1], 1..k, input1.shape[axis+1], ..., input1.shape[input1.rank - 1]]``.

*   **2**: Output tensor containing indices of the corresponding elements(values) from the first output tensor. The indices point to the location of selected values in the original input tensor. The shape of this output tensor is the same as the shape of the first output, that is ``[input1.shape[0], ..., input1.shape[axis-1], 1..k, input1.shape[axis+1], ..., input1.shape[input1.rank - 1]]``. The type of this tensor *T_IND* is controlled by the ``index_element_type`` attribute.

**Types**

* *T*: any numeric type.

* *T_IND*: ``int64`` or ``int32``.

**Detailed Description**

The output tensor is populated by values computed in the following way:

.. code-block:: cpp

   output[i1, ..., i(axis-1), j, i(axis+1) ..., iN] = top_k(input[i1, ...., i(axis-1), :, i(axis+1), ..., iN]), k, sort, mode)

meaning that for each slice ``input[i1, ...., i(axis-1), :, i(axis+1), ..., iN]`` the *TopK* values are computed individually.

Sorting and minimum/maximum are controlled by ``sort`` and ``mode`` attributes with additional configurability provided by ``stable``:

* *sort* =  ``value`` , *mode* =  ``max`` , *stable* =  ``false``  - descending by value, relative order of equal elements not guaranteed to be maintained
* *sort* = ``value`` , *mode* =  ``max`` , *stable* =  ``true``   - descending by value, relative order of equal elements guaranteed to be maintained
* *sort* = ``value`` , *mode* =  ``min`` , *stable* =  ``false``  - ascending by value, relative order of equal elements not guaranteed to be maintained
* *sort* = ``value`` , *mode* =  ``min`` , *stable* =  ``true``   - ascending by value, relative order of equal elements guaranteed to be maintained
* *sort* =  ``index`` , *mode* =  ``max`` , *stable* =  ``false``  - ascending by index, relative order of equal elements not guaranteed to be maintained
* *sort* =  ``index`` , *mode* =  ``max`` , *stable* =  ``true``   - ascending by index, relative order of equal elements guaranteed to be maintained
* *sort* =  ``index`` , *mode* =  ``min`` , *stable* =  ``false``  - ascending by index, relative order of equal elements not guaranteed to be maintained
* *sort* =  ``index`` , *mode* =  ``min`` , *stable* =  ``true``   - ascending by index, relative order of equal elements guaranteed to be maintained
* *sort* =  ``none``  , *mode* =  ``max``  - undefined
* *sort* =  ``none``  , *mode* =  ``min``  - undefined

The relative order of equivalent elements is only preserved if the ``stable`` attribute is set to ``true``. This makes the implementation use stable sorting algorithm during the computation of TopK elements. Otherwise the output order is undefined.
The "by index" order means that the input tensor's elements are still sorted by value but their order in the output tensor is additionally determined by the indices of those elements in the input tensor. This might yield multiple correct results though. For example if the input tensor contains the following elements:

.. code-block:: cpp

  input = [5, 3, 1, 2, 5, 5]

and when TopK is configured the following way:

.. code-block:: cpp

  mode = min
  sort = index
  k = 4

then the 3 following results are correct:

.. code-block:: cpp

  output_values  = [5, 3, 1, 2]
  output_indices = [0, 1, 2, 3]

  output_values  = [3, 1, 2, 5]
  output_indices = [1, 2, 3, 4]

  output_values  = [3, 1, 2, 5]
  output_indices = [1, 2, 3, 5]

When the ``stable`` attribute is additionally set to *true*, the example above will only have a single correct solution:

.. code-block:: cpp

  output_values  = [5, 3, 1, 2]
  output_indices = [0, 1, 2, 3]

The indices are always sorted ascendingly when ``sort == index`` for any given TopK node. Setting ``sort == index`` and ``mode == max`` means gthat the values are first sorted in the descending order but the indices which affect the order of output elements are sorted ascendingly.

**Example**

This example assumes that ``K`` is equal to 10:

.. code-block:: cpp

  <layer ... type="TopK" ... >
      <data axis="3" mode="max" sort="value" stable="true" index_element_type="i64"/>
      <input>
          <port id="0">
              <dim>1</dim>
              <dim>3</dim>
              <dim>224</dim>
              <dim>224</dim>
          </port>
          <port id="1">
          </port>
      <output>
          <port id="2">
              <dim>1</dim>
              <dim>3</dim>
              <dim>224</dim>
              <dim>10</dim>
          </port>
          <port id="3">
              <dim>1</dim>
              <dim>3</dim>
              <dim>224</dim>
              <dim>10</dim>
          </port>
      </output>
  </layer>


