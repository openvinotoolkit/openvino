## BinaryConvolution<a name="BinaryConvolution"></a> {#openvino_docs_ops_convolution_BinaryConvolution_1}

**Versioned name**: *BinaryConvolution-1*

**Category**: *Convolution*

**Short description**: *BinaryConvolution* convolution with binary weights, binary input and integer output

**Attributes**:

The operation has the same attributes as a regular *Convolution* layer and several unique attributes that are listed below:

* *mode*

  * **Description**: *mode* defines how input tensor 0/1 values and weights 0/1 are interpreted as real numbers and how the result is computed.
  * **Range of values**:
    * *xnor-popcount*
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *yes*

* *pad_value*

  * **Description**: *pad_value* is a floating-point value used to fill pad area.
  * **Range of values**: a floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: ND tensor with N >= 3, containing integer, float or binary values; filled with 0/1 values of any appropriate type. 0 means -1, 1 means 1 for `mode="xnor-popcount"`. Required.

*   **2**: ND tensor with N >= 3 that represents convolutional kernel filled by integer, float or binary values; filled with 0/1 values. 0 means -1, 1 means 1 for `mode="xnor-popcount"`. Required.

**Outputs**:

*   **1**: output tensor containing float values.

