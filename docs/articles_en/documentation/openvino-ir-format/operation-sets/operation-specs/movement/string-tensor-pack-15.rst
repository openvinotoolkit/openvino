.. {#openvino_docs_ops_type_StringTensorPack_15}

StringTensorPack
===================


.. meta::
  :description: Learn about StringTensorPack-15 - data movement operation which packs a concatenated batch of strings into a batched string tensor.

**Versioned name**: *StringTensorPack-15*

**Category**: *Data movement*

**Short description**: *StringTensorPack* transforms a concatenated batch of strings into 
a string tensor using *begins* and *ends* indices.

**Detailed description**

Consider inputs:

* *begins* = [0, 5]
* *ends* = [5, 13]
* *symbols* = "IntelOpenVINO"

*StringTensorPack* uses indices from ``begins`` and ``ends`` to transform concatenated string ``symbols`` into ``output``, 
a batched string tensor. The ``output.shape`` (also called ``batch_size``) is equal to ``begins.shape`` and ``ends.shape``, 
and in this case holds values ``["Intel", "OpenVINO"]``.

When defining *begins* and *ends*, the notation ``[a, b)`` is used. This means that the range starts with ``a`` and includes all values up to, 
but not including, ``b``. That is why in the example given the length of "IntelOpenVINO" is 12, but *ends* vector contains 13.

**Inputs**

* **1**: *begins*:

  * **Description**: Indices of each string's beginnings. **Required.**
  * **Shape**: 1D tensor of shape ``(batch_size)``.
  * **Type**: *T_IDX*

* **2**: *ends*:

  * **Description**: Indices of each string's endings. **Required.**
  * **Shape**: 1D tensor of shape ``(batch_size)``.
  * **Type**: *T_IDX*

* **3**: *symbols*:

  * **Description**: Concatenated ``input`` strings encoded in utf-8 bytes. **Required.**
  * **Shape**: 1D tensor of shape equal to the total length of concatenated strings.
  * **Type**: *T*

**Outputs**

* **1**: *output*

  * **Description**: A string tensor.
  * **Shape**: ``(batch_size)``
  * **Type**: *T*

**Types**

* *T*: ``u8``.
* *T_IDX*: ``int64``.

**Examples**

*Example 1: input data as string*

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
            <port id="2" precision="u8">
                <dim>13</dim>    <!-- symbols = "IntelOpenVINO" -->
            </port>
        </input>
        <output>
            <port id="0" precision="u8">
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
            <port id="2" precision="u8">
                <dim>13</dim>    <!-- symbols = "OMZGenAI 2024"-->
            </port>
        </input>
        <output>
            <port id="0" precision="u8">
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
            <port id="2" precision="u8">
                <dim>13</dim>    <!-- symbols = "123456789"-->
            </port>
        </input>
        <output>
            <port id="0" precision="u8">
                <dim>5</dim>     <!-- output = ["1", "9"] -->
            </port>
        </output>
    </layer>
