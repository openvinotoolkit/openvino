.. {#openvino_docs_ops_type_StringTensorUnpack_15}

StringTensorUnpack
===================


.. meta::
  :description: Learn about StringTensorUnpack-15 - data movement operation which unpacks a batch of strings into three tensors.

**Versioned name**: *StringTensorUnpack-15*

**Category**: *Data movement*

**Short description**: *StringTensorUnpack* operation transforms a given batch of strings into three tensors - one containing 
the concatenated string data, and two other storing begin and end indices of the strings, respectively.

**Detailed description**

Consider an ``input`` string tensor containing values ``["Intel", "OpenVINO"]``.

The operator will transform the tensor into three outputs:

* *word_begins* = [0, 5]
    * ``word_begins[0]`` is equal to 0, because the first word starts at the beggining index.
    * ``word_begins[1]`` is equal to 5, because length of the word "Intel" is equal to 5.
    * ``word_begins.shape`` is equal to [2], because the ``input`` is a batch of 2 words.

* *word_ends* = [5, 13]
    * ``word_ends[0]`` is equal to 5, because length of the word "Intel" is equal to 5.
    * ``word_ends[1]`` is equal to 13, because length of the word "OpenVINO" is 8, and it needs to be summed up
    with length of the word "Intel".
    * ``word_ends.shape`` is equal to ``[2]``, because the ``input`` is a batch of 2 words.

* *output_symbols* = "IntelOpenVINO"
    * ``output_symbols`` contains concatenated string data, interpretable using ``word_begins`` and ``word_ends``.
    * ``output_symbols.shape`` is equal to ``[13]``, because it's the length of concatenated ``input`` words.

**Inputs**

* **1**: *data*

  * **Description**: A tensor containing a string to be unpacked. **Required.**
  * **Type**: *T*

**Outputs**

* **1**: *word_begins*:

  * **Description**: Indices of each string's begginings.
  * **Shape**: 1D tensor of shape ``(batch_size)``.
  * **Type**: *T_IDX*

* **2**: *word_ends*:

  * **Description**: Indices of each string's endings.
  * **Shape**: 1D tensor of shape ``(batch_size)``.
  * **Type**: *T_IDX*

* **3**: *output_symbols*:

  * **Description**: Concatenated ``input`` words.
  * **Shape**: 1D tensor of shape equal to the total length of concatenated words.
  * **Type**: *T*

**Types**

* *T*: ``string``.
* *T_IDX*: ``int64``.

**Examples**

*Example 1: input data as string*

For ``input = ["Intel", "OpenVINO"]``

.. code-block:: xml
   :force:

    <layer ... type="StringTensorUnpack" ... >
        <input>
            <port id="0" precision="string">
                <dim>2</dim>     <!-- batch of strings -->
            </port>
        </input>
        <output>
            <port id="0" precision="I64">
                <dim>2</dim>     <!-- word_begins = [0, 5] -->
            </port>
            <port id="1" precision="I64">
                <dim>2</dim>     <!-- word_ends = [5, 13] -->
            </port>
            <port id="2" precision="string">
                <dim>13</dim>     <!-- output_symbols = "IntelOpenVINO" -->
            </port>
        </output>
    </layer>

*Example 2: input with an empty string*

For ``input = ["OMZ", "", "GenAI", " ", "2024"]``

.. code-block:: xml
   :force:

    <layer ... type="StringTensorUnpack" ... >
        <input>
            <port id="0" precision="string">
                <dim>5</dim>     <!-- batch of strings -->
            </port>
        </input>
        <output>
            <port id="0" precision="I64">
                <dim>2</dim>     <!-- word_begins = [0, 3, 3, 8, 9] -->
            </port>
            <port id="1" precision="I64">
                <dim>2</dim>     <!-- word_ends = [3, 3, 8, 9, 13] -->
            </port>
            <port id="2" precision="string">
                <dim>13</dim>    <!-- output_symbols = "OMZGenAI 2024"-->
            </port>
        </output>
    </layer>
