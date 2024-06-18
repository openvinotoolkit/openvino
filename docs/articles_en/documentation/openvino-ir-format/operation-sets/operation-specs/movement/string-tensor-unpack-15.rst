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

* *begins* = [0, 5]
    * ``begins[0]`` is equal to 0, because the first string starts at the beggining index.
    * ``begins[1]`` is equal to 5, because length of the string "Intel" is equal to 5.
    * ``begins.shape`` is equal to [2], because the ``input`` is a batch of 2 strings.

* *ends* = [5, 13]
    * ``ends[0]`` is equal to 5, because length of the string "Intel" is equal to 5.
    * ``ends[1]`` is equal to 13, because length of the string "OpenVINO" is 8, and it needs to be summed up
    with length of the string "Intel".
    * ``ends.shape`` is equal to ``[2]``, because the ``input`` is a batch of 2 strings.

* *symbols* = "IntelOpenVINO"
    * ``symbols`` contains concatenated string data, interpretable using ``begins`` and ``ends``.
    * ``symbols.shape`` is equal to ``[13]``, because it's the length of concatenated ``input`` strings.

**Inputs**

* **1**: *data*

  * **Description**: A tensor containing a string to be unpacked. **Required.**
  * **Type**: *T*

**Outputs**

* **1**: *begins*:

  * **Description**: Indices of each string's begginings.
  * **Shape**: 1D tensor of shape ``(batch_size)``.
  * **Type**: *T_IDX*

* **2**: *ends*:

  * **Description**: Indices of each string's endings.
  * **Shape**: 1D tensor of shape ``(batch_size)``.
  * **Type**: *T_IDX*

* **3**: *symbols*:

  * **Description**: Concatenated ``input`` strings.
  * **Shape**: 1D tensor of shape equal to the total length of concatenated string.
  * **Type**: *T*

**Types**

* *T*: ``u8``.
* *T_IDX*: ``int64``.

**Examples**

*Example 1: input data as string*

For ``input = ["Intel", "OpenVINO"]``

.. code-block:: xml
   :force:

    <layer ... type="StringTensorUnpack" ... >
        <input>
            <port id="0" precision="u8">
                <dim>2</dim>     <!-- batch of strings -->
            </port>
        </input>
        <output>
            <port id="0" precision="I64">
                <dim>2</dim>     <!-- begins = [0, 5] -->
            </port>
            <port id="1" precision="I64">
                <dim>2</dim>     <!-- ends = [5, 13] -->
            </port>
            <port id="2" precision="u8">
                <dim>13</dim>     <!-- symbols = "IntelOpenVINO" -->
            </port>
        </output>
    </layer>

*Example 2: input with an empty string*

For ``input = ["OMZ", "", "GenAI", " ", "2024"]``

.. code-block:: xml
   :force:

    <layer ... type="StringTensorUnpack" ... >
        <input>
            <port id="0" precision="u8">
                <dim>5</dim>     <!-- batch of strings -->
            </port>
        </input>
        <output>
            <port id="0" precision="I64">
                <dim>2</dim>     <!-- begins = [0, 3, 3, 8, 9] -->
            </port>
            <port id="1" precision="I64">
                <dim>2</dim>     <!-- ends = [3, 3, 8, 9, 13] -->
            </port>
            <port id="2" precision="u8">
                <dim>13</dim>    <!-- symbols = "OMZGenAI 2024"-->
            </port>
        </output>
    </layer>
