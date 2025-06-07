# SparseFillEmptyRowsUnpackedString

## Meta
```
.. meta::
  :description: Learn about SparseFillEmptyRowsUnpackedString-16 - a sparse operation that fills empty rows 
                in a sparse string tensor with a default string value.
```

**Versioned name**: *SparseFillEmptyRowsUnpackedString-16*

**Category**: *Sparse*

**Short description**: Fills empty rows of an input sparse string tensor with a default string value.

**Detailed description**:

Operation SparseFillEmptyRowsUnpackedString works with string tensor data represented in the unpacked format (begins, ends, symbols). It is similar to SparseFillEmptyRows but processes string data instead of numeric values.

The input sparse string tensor is represented by the inputs:
* `begins`
* `ends`
* `symbols`

The dense shape of the tensor is inferred from the shape of the `begins` and `ends` tensors.

For each row in the input sparse string tensor, this operator checks if the row is empty. A row is considered empty if it contains only zero-length strings (where begins[i] == ends[i] for all entries in that row). If the row is empty, the operator adds an entry with the specified default string value at index `[row, 0]`.

The output tensor will have updated `output_begins`, `output_ends`, and `output_symbols`.

This operator also returns a boolean vector indicating which rows were filled with the default value: `empty_row_indicator[i] = True` if row `i` was an empty row.

**Attributes**: SparseFillEmptyRowsUnpackedString-16 operation has no attributes.

**Inputs**:

* **1**: `begins` - ND tensor of type *T_IDX* containing the beginning indices of strings in the `symbols` array. **Required.**
* **2**: `ends` - ND tensor of type *T_IDX* containing the ending indices of strings in the `symbols` array. The shape must be identical to `begins`. **Required.**
* **3**: `symbols` - 1D tensor of type *u8* containing the concatenated string data encoded in utf-8 bytes. **Required.**
* **4**: `default_value` - A 1D tensor of type *u8* describing a string to be inserted into the empty rows. **Required.**

**Outputs**:

* **1**: `output_begins` - ND tensor of type *T_IDX* containing the beginning indices of strings in the `output_symbols` array.
* **2**: `output_ends` - ND tensor of type *T_IDX* containing the ending indices of strings in the `output_symbols` array.
* **3**: `output_symbols` - 1D tensor of type *u8* containing the concatenated string data encoded in utf-8 bytes.
* **4**: `empty_row_indicator` - 1D tensor of type `boolean` indicating True for rows which were empty before executing the operation.

**Types**

* *T_IDX*: `int32` or `int64`.

**Example**

*Example: sparse string tensor with shape [5, 2].*

Input:
* `begins = [[0, 5], [10, 10], [15, 20], [25, 30], [35, 35]]` (note row 1 and row 4 have empty strings)
* `ends = [[5, 10], [10, 10], [20, 25], [30, 35], [35, 35]]`
* `symbols = "HelloWorldOpenVINOTensorProcessing"` (encoded in utf-8 bytes)
* `default_value = "Empty"` (encoded in utf-8 bytes)

Rows 1 and 4 are empty. The output will be:

* `output_begins = [[0, 5], [40, 10], [15, 20], [25, 30], [45, 35]]`
* `output_ends = [[5, 10], [45, 10], [20, 25], [30, 35], [50, 35]]`
* `output_symbols = "HelloWorldEmptyOpenVINOTensorProcessingEmpty"` (encoded in utf-8 bytes)
* `empty_row_indicator = [False, True, False, False, True]`

```xml
<layer ... type="SparseFillEmptyRowsUnpackedString" ... >
    <input>
        <port id="0" precision="I32">       <!-- begins -->
            <dim>5</dim>
            <dim>2</dim>
        </port>
        <port id="1" precision="I32">       <!-- ends -->
            <dim>5</dim>
            <dim>2</dim>
        </port>
        <port id="2" precision="U8">        <!-- symbols -->
            <dim>35</dim>
        </port>
        <port id="3" precision="U8">        <!-- default_value -->
            <dim>5</dim>
        </port>
    </input>
    <output>
        <port id="4" precision="I32">       <!-- output_begins -->
            <dim>5</dim>
            <dim>2</dim>
        </port>
        <port id="5" precision="I32">       <!-- output_ends -->
            <dim>5</dim>
            <dim>2</dim>
        </port>
        <port id="6" precision="U8">        <!-- output_symbols -->
            <dim>50</dim>
        </port>
        <port id="7" precision="BOOL">      <!-- empty_row_indicator -->
            <dim>5</dim>
        </port>
    </output>
</layer>
```
