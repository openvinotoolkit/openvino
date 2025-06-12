Unpacked String Tensors
==========================================


.. meta::
  :description: Learn about OpenVINO's Unpacked String Tensor formats used for efficient handling of string data in neural network operations.

The purpose of this document is to provide specifications for Unpacked String Tensor formats in OpenVINO, which are used for efficient processing of string data.

Description
###########

OpenVINO supports two formats for representing string data in an unpacked manner: the `UnpackedStringTensor` format and the `SparseUnpackedStringTensor` format.

`UnpackedStringTensor` Format
*****************************

The `UnpackedStringTensor` format represents a string tensor as three separate tensors:

1. **begins**: ND tensor of non-negative integer numbers containing the starting indices of each string in the `symbols` array.
2. **ends**: ND tensor of non-negative integer numbers containing the ending indices of each string in the `symbols` array.
3. **symbols**: 1D tensor of type *u8* containing the concatenated string data encoded in utf-8 bytes.

The `begins` and `ends` tensors have the same shape, which matches the shape of the original string tensor.

When defining *begins* and *ends*, the notation ``[a, b)`` is used. This means that the range starts with ``a`` and includes all values up to, but not including, ``b``. 

The `UnpackedStringTensor` format can be produced from regular string tensors using the `StringTensorUnpack` operation, and it can be converted back to a regular string tensor using the `StringTensorPack` operation.

**Examples**:

*Example 1: 1D string tensor*

For a string tensor ``["Intel", "OpenVINO"]``, the unpacked representation would be:

* *begins* = [0, 5]
* *ends* = [5, 13]
* *symbols* = "IntelOpenVINO"

*Example 2: 2D string tensor*

For a string tensor ``[["Intel", "OpenVINO"], ["OMZ", "GenAI"]]``, the unpacked representation would be:

* *begins* = [[0, 5], [13, 16]]
* *ends* = [[5, 13], [16, 21]]
* *symbols* = "IntelOpenVINOOMZGenAI"

*Example 3: String tensor with an empty string*

For a string tensor ``["OMZ", "", "GenAI", " ", "2024"]``, the unpacked representation would be:

* *begins* = [0, 3, 3, 8, 9]
* *ends* = [3, 3, 8, 9, 13]
* *symbols* = "OMZGenAI 2024"

`SparseUnpackedStringTensor` Format
************************************

The `SparseUnpackedStringTensor` format extends the `UnpackedStringTensor` format by adding sparse tensor representation, where no redundant indices in `begins` and `ends` tensors are reserved for empty strings. It consists of five tensors:

1. **begins**: 1D tensor of type containing the beginning indices of strings in the `symbols` array.
2. **ends**: 1D tensor of type containing the ending indices of strings in the `symbols` array.
3. **symbols**: 1D tensor of type *u8* containing the concatenated string data encoded in utf-8 bytes.
4. **indices**: 2D tensor of type indicating the positions at which values are placed in the sparse tensor.
5. **dense_shape**: 1D tensor of type indicating the shape of the dense tensor.

The `begins` and `ends` tensors have the same length, which matches the first dimension of the `indices` tensor.

**Example**:

For a sparse string tensor with these properties:

* Non-zero elements at positions `[(0,0), (0,1), (2,0), (2,1), (3,0), (3,1)]` in a tensor of shape `[5, 2]`
* Values: "Hello", "World", "Open", "VINO", "Tensor", "Processing"

The sparse unpacked representation would be:

* *begins* = [0, 5, 10, 14, 18, 25]
* *ends* = [5, 10, 14, 18, 25, 34]
* *symbols* = "HelloWorldOpenVINOTensorProcessing"
* *indices* = [[0, 0], [0, 1], [2, 0], [2, 1], [3, 0], [3, 1]]
* *dense_shape* = [5, 2]
