StringTensorUnpack
===================


.. meta::
  :description: Learn about StringTensorUnpack-15 - operation which unpacks a batch of strings into three tensors.

**Versioned name**: *StringTensorUnpack-15*

**Category**: *Type*

**Short description**: *StringTensorUnpack* operation transforms a given batch of strings into three tensors - two storing begin
and end indices of the strings and another containing the concatenated string data, respectively.

**Detailed description**

Consider an ``input`` string tensor containing values ``["Intel", "OpenVINO"]``.

The operator will transform the tensor into three outputs:

* *begins* = [0, 5]
    * ``begins[0]`` is equal to 0, because the first string starts at the beginning index.
    * ``begins[1]`` is equal to 5, because length of the string "Intel" is equal to 5.
    * ``begins.shape`` is equal to [2], because the ``input`` is a batch of 2 strings.

* *ends* = [5, 13]
    * ``ends[0]`` is equal to 5, because length of the string "Intel" is equal to 5.
    * ``ends[1]`` is equal to 13, because length of the string "OpenVINO" is 8, and it needs to be summed up with length of the string "Intel".
    * ``ends.shape`` is equal to ``[2]``, because the ``input`` is a batch of 2 strings.

* *symbols* = "IntelOpenVINO"
    * ``symbols`` contains concatenated string data encoded in utf-8 bytes, interpretable using ``begins`` and ``ends``.
    * ``symbols.shape`` is equal to ``[13]``, because it's the length of concatenated ``input`` strings.

When defining *begins* and *ends*, the notation ``[a, b)`` is used. This means that the range starts with ``a`` and includes all values up to,
but not including, ``b``. That is why in the example given the length of "IntelOpenVINO" is 12, but *ends* vector contains 13.

**Inputs**

* **1**: ``data`` - ND tensor of type *string*. **Required.**

**Outputs**

* **1**: ``begins`` - ND tensor of non-negative integer numbers of type *int32* and of the same shape as ``data`` input.

* **2**: ``ends`` - ND tensor of non-negative integer numbers of type *int32* and of the same shape as ``data`` input.

* **3**: ``symbols`` - 1D tensor of concatenated strings data encoded in utf-8 bytes, of type *u8* and size equal to the sum of the lengths of each string from the ``data`` input.

**Examples**

*Example 1: 1D input*

For ``input = ["Intel", "OpenVINO"]``

.. code-block:: xml
   :force:

    <layer ... type="StringTensorUnpack" ... >
        <input>
            <port id="0" precision="STRING">
                <dim>2</dim>     <!-- batch of strings -->
            </port>
        </input>
        <output>
            <port id="0" precision="I32">
                <dim>2</dim>     <!-- begins = [0, 5] -->
            </port>
            <port id="1" precision="I32">
                <dim>2</dim>     <!-- ends = [5, 13] -->
            </port>
            <port id="2" precision="U8">
                <dim>13</dim>     <!-- symbols = "IntelOpenVINO" encoded in an utf-8 array -->
            </port>
        </output>
    </layer>

*Example 2: input with an empty string*

For ``input = ["OMZ", "", "GenAI", " ", "2024"]``

.. code-block:: xml
   :force:

    <layer ... type="StringTensorUnpack" ... >
        <input>
            <port id="0" precision="STRING">
                <dim>5</dim>     <!-- batch of strings -->
            </port>
        </input>
        <output>
            <port id="0" precision="I32">
                <dim>2</dim>     <!-- begins = [0, 3, 3, 8, 9] -->
            </port>
            <port id="1" precision="I32">
                <dim>2</dim>     <!-- ends = [3, 3, 8, 9, 13] -->
            </port>
            <port id="2" precision="U8">
                <dim>13</dim>    <!-- symbols = "OMZGenAI 2024" encoded in an utf-8 array -->
            </port>
        </output>
    </layer>

*Example 3: 2D input*

For ``input = [["Intel", "OpenVINO"], ["OMZ", "GenAI"]]``

.. code-block:: xml
   :force:

    <layer ... type="StringTensorUnpack" ... >
        <input>
            <port id="0" precision="STRING">
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </input>
        <output>
            <port id="0" precision="I32">
                <dim>2</dim>     <!-- begins = [[0, 5], [13, 16]] -->
                <dim>2</dim>
            </port>
            <port id="1" precision="I32">
                <dim>2</dim>     <!-- ends = [[5, 13], [16, 21]] -->
                <dim>2</dim>
            </port>
            <port id="2" precision="U8">
                <dim>21</dim>    <!-- symbols = "IntelOpenVINOOMZGenAI" encoded in an utf-8 array -->
            </port>
        </output>
    </layer>
