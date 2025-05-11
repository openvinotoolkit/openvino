ReverseSequence
===============


.. meta::
  :description: Learn about ReverseSequence-1 - a data movement operation,
                which can be performed on two required input tensors.

**Versioned name**: *ReverseSequence-1*

**Category**: *Data movement*

**Short description**: *ReverseSequence* reverses variable length slices of data.

**Detailed description**

*ReverseSequence* slices a given input tensor ``data`` along the dimension specified in the *batch_axis* attribute. For each slice ``i``, it reverses the first ``seq_lengths[i]`` elements along the dimension specified in the *seq_axis* attribute.

**Attributes**

* *batch_axis*

  * **Description**: *batch_axis* is the index of the batch dimension along which ``data`` input tensor is sliced.
  * **Range of values**: an integer within the range ``[-rank(data), rank(data) - 1]``
  * **Type**: ``int``
  * **Default value**: ``0``
  * **Required**: *no*

* *seq_axis*

  * **Description**: *seq_axis* is the index of the sequence dimension along which elements of ``data`` input tensor are reversed.
  * **Range of values**: an integer within the range ``[-rank(data), rank(data) - 1]``
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - Input data to reverse. A tensor of type *T1* and rank greater or equal to 2. **Required.**
* **2**: ``seq_lengths`` - Sequence lengths to reverse in the input tensor ``data``. A 1D tensor comprising ``data_shape[batch_axis]`` elements of type *T2*. All element values must be integer values within the range ``[1, data_shape[seq_axis]]``. Value ``1`` means, no elements are reversed. **Required.**

**Outputs**

* **1**: The result of slice and reverse ``data`` input tensor. A tensor of type *T1* and the same shape as ``data`` input tensor.

**Types**

* *T1*: any supported type.
* *T2*: any supported numerical type.

**Example**

.. code-block:: xml
   :force:

    <layer ... type="ReverseSequence">
        <data batch_axis="0" seq_axis="1"/>
        <input>
            <port id="0">       <!-- data -->
                <dim>4</dim>    <!-- batch_axis -->
                <dim>10</dim>   <!-- seq_axis -->
                <dim>100</dim>
                <dim>200</dim>
            </port>
            <port id="1">
                <dim>4</dim>    <!-- seq_lengths value: [2, 4, 8, 10] -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>4</dim>
                <dim>10</dim>
                <dim>100</dim>
                <dim>200</dim>
            </port>
        </output>
    </layer>



