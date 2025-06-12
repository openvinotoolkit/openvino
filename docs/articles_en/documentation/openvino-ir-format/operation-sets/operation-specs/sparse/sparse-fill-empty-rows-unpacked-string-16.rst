# SparseFillEmptyRowsUnpackedString

.. meta::
  :description: Learn about SparseFillEmptyRowsUnpackedString-16 - a sparse operation that fills empty rows 
                in a `SparseUnpackedString` tensor with a default string value.

**Versioned name**: *SparseFillEmptyRowsUnpackedString-16*

**Category**: *Sparse*

**Short description**: Fills empty rows of an input sparse string tensor with a default string value.

**Detailed description**:

Operation `SparseFillEmptyRowsUnpackedString` works with string tensor data represented in the `SparseUnpackedString` Tensor format. For detailed information about this format, see the :doc:`Unpacked String Tensor Formats <../../unpacked-string-tensors>` specification.

For each row in the input sparse tensor, this operator checks if the row is empty. If the row is empty, the operator adds an entry with the specified default string value at index `[row, 0]`. The output will have updated indices and values in the `SparseUnpackedString` Tensor format.

This operator also returns a boolean vector indicating which rows were filled with the default value: `empty_row_indicator[i] = True` if row `i` was an empty row.

**Attributes**: SparseFillEmptyRowsUnpackedString-16 operation has no attributes.

**Inputs**:

* **1**: `begins` - 1D tensor of type *T_IDX* containing the beginning indices of strings in the `symbols` array. **Required.**
* **2**: `ends` - 1D tensor of type *T_IDX* containing the ending indices of strings in the `symbols` array. The length must be identical to `begins`. **Required.**
* **3**: `symbols` - 1D tensor of type *u8* containing the concatenated string data encoded in utf-8 bytes. **Required.**
* **4**: `indices` - 2D tensor of type *T_IDX* and non-negative values indicating the positions at which values are placed in the sparse tensor. It is of shape `[M, 2]`, where `M` is the same as the length of the `begins` and `ends` inputs. **Required.**
* **5**: `dense_shape` - 1D tensor of type *T_IDX* indicating the shape of the dense tensor. **Required.**
* **6**: `default_value` - A 1D tensor of type *u8* describing a string to be inserted into the empty rows. **Required.**

**Outputs**:

* **1**: `output_begins` - 1D tensor of type *T_IDX* containing the beginning indices of strings in the `output_symbols` array. Shape is `[M']`, where `M'` is the number of entries in the output sparse tensor.
* **2**: `output_ends` - 1D tensor of type *T_IDX* containing the ending indices of strings in the `output_symbols` array. Shape is `[M']`, where `M'` is the number of entries in the output sparse tensor.
* **3**: `output_indices` - 2D tensor of type *T_IDX* indicating the positions at which values are placed in the sparse tensor. It is of shape `[M', 2]`, where `M'` is the number of entries in the output sparse tensor.
* **4**: `output_symbols` - 1D tensor of type *u8* containing the concatenated string data encoded in utf-8 bytes. It is extended with the default value if any empty rows are found. Shape is `[S']`, where `S'` is the total length of all strings concatenated.
* **5**: `empty_row_indicator` - 1D tensor of type `boolean` indicating True for rows which were empty before executing the operation. Shape is `[dense_shape[0]]`.

**Types**

* *T_IDX*: `int32` or `int64`.

**Example**

*Example: sparse string tensor with shape [5, 2].*

Input:

* `begins = [0, 5, 15, 20, 25, 30]`
* `ends = [5, 10, 20, 25, 30, 35]`
* `symbols = "HelloWorldOpenVINOTensorProcessing"` (encoded in utf-8 bytes)
* `indices = [[0, 0], [0, 1], [2, 0], [2, 1], [3, 0], [3, 1]]`
* `dense_shape = [5, 2]`
* `default_value = "Empty"` (encoded in utf-8 bytes)

Rows 1 and 4 are empty. The output will be:

* `output_begins = [0, 5, 35, 15, 20, 25, 30, 35]`
* `output_ends = [5, 10, 40, 20, 25, 30, 35, 40]`
* `output_indices = [[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0]]`
* `output_symbols = "HelloWorldOpenVINOTensorProcessingEmpty"` (encoded in utf-8 bytes)
* `empty_row_indicator = [False, True, False, False, True]`

`default_value` is stored only once in the `output_symbols`, and the `output_begins` and `output_ends` are adjusted accordingly to include the default value for the empty rows. In this particular case, the `default_value` is represented by indices 35 to 40.

```xml
<layer ... type="SparseFillEmptyRowsUnpackedString" ... >
    <input>
        <port id="0" precision="I32">       <!-- begins -->
            <dim>6</dim>
        </port>
        <port id="1" precision="I32">       <!-- ends -->
            <dim>6</dim>
        </port>
        <port id="2" precision="U8">        <!-- symbols -->
            <dim>35</dim>
        </port>
        <port id="3" precision="I32">       <!-- indices -->
            <dim>6</dim>
            <dim>2</dim>
        </port>
        <port id="4" precision="I32">       <!-- dense_shape -->
            <dim>2</dim>
        </port>
        <port id="5" precision="U8">        <!-- default_value -->
            <dim>5</dim>
        </port>
    </input>
    <output>
        <port id="0" precision="I32">       <!-- output_begins -->
            <dim>8</dim>
        </port>
        <port id="1" precision="I32">       <!-- output_ends -->
            <dim>8</dim>
        </port>
        <port id="2" precision="I32">       <!-- output_indices -->
            <dim>8</dim>
            <dim>2</dim>
        </port>
        <port id="3" precision="U8">        <!-- output_symbols -->
            <dim>40</dim>
        </port>
        <port id="4" precision="BOOL">      <!-- empty_row_indicator -->
            <dim>5</dim>
        </port>
    </output>
</layer>
```
