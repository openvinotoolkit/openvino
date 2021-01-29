## BinaryConvolution<a name="BinaryConvolution"></a> {#openvino_docs_ops_convolution_BinaryConvolution_1}

**Versioned name**: *BinaryConvolution-1*

**Category**: *Convolution*

**Short description**: Computes 1D, 2D or 3D convolution of binary input and binary kernel tensors.

**Detailed description**: The operation bahaves as regular *Convolution* but uses specialized algorithm for computations on binary data. More thorough explanation can be found in [Understanding Binary Neural Networks](https://sushscience.wordpress.com/2017/10/01/understanding-binary-neural-networks/) and [Bitwise Neural Networks](https://saige.sice.indiana.edu/wp-content/uploads/icml2015_mkim.pdf).  

**Attributes**:

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the `(z, y, x)` axes for 3D convolutions and `(y, x)` axes for 2D convolutions. For example, *strides* equal `4,2,1` means sliding the filter 4 pixel at a time over depth dimension, 2 over height dimension and 1 over width dimension.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal `1,2` means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal `1,2` means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* denotes the distance in width and height between elements (weights) in the filter. For example, *dilation* equal `1,1` means that all the elements in the filter are neighbors, so it is the same as for the usual convolution. *dilation* equal `2,2` means that all the elements in the filter are matched not to adjacent elements in the input matrix, but to those that are adjacent with distance 1.
  * **Range of values**: integer value starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *mode*

  * **Description**: *mode* defines how input tensor `0/1` values and weights `0/1` are interpreted as real numbers and how the result is computed.
  * **Range of values**:
    * *xnor-popcount*
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *yes*
  *  **Note**: value `0` in inputs is interpreted as `-1`, value `1` as `1`

* *pad_value*

  * **Description**: *pad_value* is a floating-point value used to fill pad area.
  * **Range of values**: a floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:
    * *explicit* - use explicit padding values from *pads_begin* and *pads_end*.
    * *same_upper* - the input is padded to match the output size. In case of odd padding value an extra padding is added at the end.
    * *same_lower* - the input is padded to match the output size. In case of odd padding value an extra padding is added at the beginning.
    * *valid* - do not use padding.
  * **Type**: string
  * **Default value**: explicit
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.
  
**Inputs**:

*   **1**: Input tensor of type *T* and rank 3, 4 or 5. Layout is NCZYX (number of batches, number of channels, spatial axes Z, Y, X). Required.
*   **2**: Kernel tensor of type *T* and rank 3, 4 or 5. Layout is OIZYX (number of output channels, number of input channels, spatial axes Z, Y, X). Required.
*   **Note**: Interpretation of tensor values is defined by *mode* attribute.
*   **Note**: Type of the convolution (1D, 2D or 3D) is derived from the rank of the input tensors and not specified by any attribute:
      * 1D convolution (input tensors rank 3) means that there is only one spatial axis X
      * 2D convolution (input tensors rank 4) means that there are two spatial axes Y, X
      * 3D convolution (input tensors rank 5) means that there are three spatial axes Z, Y, X

**Outputs**:

*   **1**: Output tensor of type *T* and rank 3, 4 or 5. Layout is NOZYX (number of batches, number of kernel output channels, spatial axes Z, Y, X).
  
**Types**:

* *T*: any floating point type with values `0` or `1`.

**Example**:

1D Convolution
```xml
<layer type="BinaryConvolution" ...>
    <data dilations="1" pads_begin="0" pads_end="0" strides="2" mode="xnor-popcount" auto_pad="valid"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>5</dim>
            <dim>128</dim>
        </port>
        <port id="1">
            <dim>16</dim>
            <dim>5</dim>
            <dim>4</dim>
        </port>
    </input>
    <output>
        <port id="2" precision="FP32">
            <dim>1</dim>
            <dim>16</dim>
            <dim>63</dim>
        </port>
    </output>
</layer>
```
2D Convolution
```xml
<layer type="BinaryConvolution" ...>
    <data dilations="1,1" pads_begin="2,2" pads_end="2,2" strides="1,1" mode="xnor-popcount" auto_pad="explicit"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
        <port id="1">
            <dim>64</dim>
            <dim>3</dim>
            <dim>5</dim>
            <dim>5</dim>
        </port>
    </input>
    <output>
        <port id="2" precision="FP32">
            <dim>1</dim>
            <dim>64</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </output>
</layer>
```

3D Convolution
```xml
<layer type="BinaryConvolution" ...>
    <data dilations="2,2,2" pads_begin="0,0,0" pads_end="0,0,0" strides="3,3,3" mode="xnor-popcount" auto_pad="explicit"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>7</dim>
            <dim>320</dim>
            <dim>320</dim>
            <dim>320</dim>
        </port>
        <port id="1">
            <dim>32</dim>
            <dim>7</dim>
            <dim>3</dim>
            <dim>3</dim>
            <dim>3</dim>
        </port>
    </input>
    <output>
        <port id="2" precision="FP32">
            <dim>1</dim>
            <dim>32</dim>
            <dim>106</dim>
            <dim>106</dim>
            <dim>106</dim>
        </port>
    </output>
</layer>
```
