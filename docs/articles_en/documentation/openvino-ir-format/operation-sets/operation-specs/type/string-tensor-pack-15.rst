StringTensorPack
===================


.. meta::
  :description: Learn about StringTensorPack-15 - operation which packs a concatenated batch of strings into a batched string tensor.

**Versioned name**: *StringTensorPack-15*

**Category**: *Type*

**Short description**: *StringTensorPack* transforms a concatenated strings data (encoded as 1D tensor of u8 element type) into
a string tensor using *begins* and *ends* indices.

**Detailed description**

Consider inputs:

* *begins* = [0, 5]
* *ends* = [5, 13]
* *symbols* = "IntelOpenVINO"

*StringTensorPack* uses indices from ``begins`` and ``ends`` to transform concatenated string ``symbols`` into ``output``,
a string tensor. The ``output.shape`` is equal to ``begins.shape`` and ``ends.shape``,
and in this case ``output`` holds values ``["Intel", "OpenVINO"]``.

When defining *begins* and *ends*, the notation ``[a, b)`` is used. This means that the range starts with ``a`` and includes all values up to,
but not including, ``b``. That is why in the example given the length of "IntelOpenVINO" is 12, but *ends* vector contains 13. The shapes of ``begins`` and ``ends`` are required to be equal.

**Inputs**

* **1**: ``begins`` - ND tensor of non-negative integer numbers of type *T_IDX*, containing indices of each string's beginnings. **Required.**

* **2**: ``ends`` - ND tensor of non-negative integer numbers of type *T_IDX*, containing indices of each string's endings. **Required.**

* **3**: ``symbols`` - 1D tensor of concatenated strings data encoded in utf-8 bytes, of type *u8*. **Required.**

**Outputs**

* **1**: ``output`` - ND string tensor of the same shape as *begins* and *ends*.

**Types**

* *T_IDX*: ``int32`` or ``int64``.

**Examples**

*Example 1: 1D begins and ends*

.. code-block:: xml
   :force:

    <layer ... type="StringTensorPack" ... >
        <input>
            <port id="0" precision="I64">
                <dim>2</dim>     <!-- begins = [0, 5] -->
            </port>
            <port id="1" precision="I64">
                <dim>2</dim>     <!-- ends = [5, 13] -->
            </port>
            <port id="2" precision="U8">
                <dim>13</dim>    <!-- symbols = "IntelOpenVINO" encoded in an utf-8 array -->
            </port>
        </input>
        <output>
            <port id="0" precision="STRING">
                <dim>2</dim>     <!-- output = ["Intel", "OpenVINO"] -->
            </port>
        </output>
    </layer>

*Example 2: input with an empty string*

.. code-block:: xml
   :force:

    <layer ... type="StringTensorPack" ... >
        <input>
            <port id="0" precision="I64">
                <dim>2</dim>     <!-- begins = [0, 3, 3, 8, 9] -->
            </port>
            <port id="1" precision="I64">
                <dim>2</dim>     <!-- ends = [3, 3, 8, 9, 13] -->
            </port>
            <port id="2" precision="U8">
                <dim>13</dim>    <!-- symbols = "OMZGenAI 2024" encoded in an utf-8 array -->
            </port>
        </input>
        <output>
            <port id="0" precision="STRING">
                <dim>5</dim>     <!-- output = ["OMZ", "", "GenAI", " ", "2024"] -->
            </port>
        </output>
    </layer>

*Example 3: skipped symbols*

.. code-block:: xml
   :force:

    <layer ... type="StringTensorPack" ... >
        <input>
            <port id="0" precision="I64">
                <dim>2</dim>     <!-- begins = [0, 8] -->
            </port>
            <port id="1" precision="I64">
                <dim>2</dim>     <!-- ends = [1, 9] -->
            </port>
            <port id="2" precision="U8">
                <dim>9</dim>     <!-- symbols = "123456789" encoded in an utf-8 array -->
            </port>
        </input>
        <output>
            <port id="0" precision="STRING">
                <dim>5</dim>     <!-- output = ["1", "9"] -->
            </port>
        </output>
    </layer>

*Example 4: 2D begins and ends*

.. code-block:: xml
   :force:

    <layer ... type="StringTensorPack" ... >
        <input>
            <port id="0" precision="I64">
                <dim>2</dim>     <!-- begins = [[0, 5], [13, 16]] -->
                <dim>2</dim>
            </port>
            <port id="1" precision="I64">
                <dim>2</dim>     <!-- ends = [[5, 13], [16, 21]] -->
                <dim>2</dim>
            </port>
            <port id="2" precision="U8">
                <dim>21</dim>    <!-- symbols = "IntelOpenVINOOMZGenAI" -->
            </port>
        </input>
        <output>
            <port id="0" precision="STRING">
                <dim>2</dim>     <!-- output = [["Intel", "OpenVINO"], ["OMZ", "GenAI"]] -->
                <dim>2</dim>
            </port>
        </output>
    </layer>
