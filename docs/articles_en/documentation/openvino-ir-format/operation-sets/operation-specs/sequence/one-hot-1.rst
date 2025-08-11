OneHot
======


.. meta::
  :description: Learn about OneHot-1 - a sequence processing operation, which
                can be performed on four required input tensors.

**Versioned name**: *OneHot-1*

**Category**: *Sequence processing*

**Short description**: *OneHot* sets the elements in the output tensor with specified indices to ``on_value`` and fills all other locations with ``off_value``.

**Detailed description**

Taking a tensor with rank ``N`` as the first input ``indices``, OneHot produces a tensor with rank ``N+1`` extending the original
tensor with a new dimension at the ``axis`` position. The output tensor is populated with two scalar values: ``on_value``
that comes from the 3rd input and ``off_value`` that comes from the 4nd input. The population is made in the following way:

.. code-block:: cpp

    output[:, ... ,:, i, :, ... ,:] = on_value if (indices[:, ..., :, :, ..., :] == i) else off_value

where ``i`` is at the ``axis`` position in the ``output`` shape and has values from the range ``[0, ..., depth-1]``.

When some elements from the ``indices`` are greater or equal to the ``depth``, it is a well-formed operation. The corresponding output rows are populated with ``off_value`` in this case.

The types of input scalars ``on_value`` and ``off_value`` should match and be equal to any supported type. The output tensor type is derived from the ``on_value`` or the ``off_value``, they all have the same type.

**Attributes**:

* *axis*

  * **Description**: *axis* is a new axis position in the output shape to fill with one-hot values.
  * **Range of values**: an integer. Negative value means counting dimension from the end.
  * **Type**: ``int``
  * **Required**: *yes*

* *negative_indices_mode*

  * **Description**: controls how negative indices in indices tensor are handled.
  * **Range of values**: 

    * ``ignore-negative``: negative indices are ignored, and the corresponding output rows are filled with ``off_value``.
    * ``normalize``: negative indices in the range ``[-depth, -1]`` are normalized by ``depth + index``; which effectively maps the range ``[-depth, depth-1]`` to ``[0, depth-1]``.

  * **Type**: ``string``
  * **Default value**: ignore-negative
  * **Required**: *no*

.. note::
   Behavior before 2025.4 OpenVINO release: negative_indices_mode was not supported and negative indices were handled according to ignore-negative mode.

**Inputs**:

* **1**: ``indices``: input tensor of type *T1* with indices. Can be 0D. **Required.**
* **2**: ``depth``: positive scalar (0D tensor) of type *T1* that specifies the number of classes and thus the size of the one-hot dimension. **Required.**
* **3**: ``on_value``: scalar (0D tensor) of type *T2* that fills the locations in output tensor specified in ``indices``. **Required.**
* **4**: ``off_value``: scalar (0D tensor) of type *T2* that fills the locations not represented in ``indices``. **Required.**

**Outputs**:

* **1**: An ``N+1`` rank tensor of type *T2*, where ``N`` is a rank of the input tensor ``indices``. A new axis of the size ``depth`` is inserted at the dimension ``axis``.

**Types**

* *T1*: ``int32`` or ``int64``.

* *T2*: any supported data type.

**Examples**

.. code-block:: xml
   :force:

    <layer ... type="OneHot" ...>
        <data axis="-1" negative_indices_mode="normalize"/>
        <input>
            <port id="0">    <!-- indices value: [0, -5, -2, 2] -->
                <dim>4</dim>
            </port>
            <port id="1">    <!-- depth value: 3 -->
            </port>
            <port id="2">    <!-- on_value 1 -->
            </port>
            <port id="3">    <!-- off_value 2 -->
            </port>
        </input>
        <output>
            <port id="0">    <!-- output value # [[1, 2, 2], [2, 2, 2], [2, 1, 2], [2, 2, 1]] -->
                <dim>4</dim>
                <dim>3</dim>
            </port>
        </output>
    </layer>



.. code-block:: xml
   :force:

    <layer ... type="OneHot" ...>
        <data axis="1"/>
        <input>
            <port id="0">    <!-- indices value: [[0, 3, 1], [1, 2, 4]] -->
                <dim>2</dim>
                <dim>3</dim>
            </port>
            <port id="1">    <!-- depth value: 3 -->
            </port>
            <port id="2">    <!-- on_value 1 -->
            </port>
            <port id="3">    <!-- off_value 0 -->
            </port>
        </input>
        <output>
            <port id="0">    <!-- output value: [[[1, 0, 0], [0, 0, 1], [0, 0, 0]], -->
                <dim>2</dim> <!--                [[0, 0, 0], [1, 0, 0], [0, 1, 0]]] -->
                <dim>3</dim>
                <dim>3</dim>
            </port>
        </output>
    </layer>



