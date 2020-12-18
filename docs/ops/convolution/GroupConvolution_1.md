## GroupConvolution <a name="GroupConvolution"></a> {#openvino_docs_ops_convolution_GroupConvolution_1}

**Versioned name**: *GroupConvolution-1*

**Category**: Convolution

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/convolution.html)

**Detailed description**: [Reference](http://cs231n.github.io/convolutional-networks/#conv)

**Attributes**

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the (z, y, x) axes for 3D convolutions and (y, x) axes for 2D convolutions. For example, *strides* equal *4,2,1* means sliding the filter 4 pixel at a time over depth dimension, 2 over height dimension and 1 over width dimension.
  * **Range of values**: positive integer numbers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal *1,2* means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: positive integer numbers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified. 

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal *1,2* means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: positive integer numbers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified. 

* *dilations*

  * **Description**: *dilations* denotes the distance in width and height between elements (weights) in the filter. For example, *dilation* equal *1,1* means that all the elements in the filter are neighbors, so it is the same as for the usual convolution. *dilation* equal *2,2* means that all the elements in the filter are matched not to adjacent elements in the input matrix, but to those that are adjacent with distance 1.
  * **Range of values**: positive integer numbers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:
    * *explicit*: use explicit padding values from `pads_begin` and `pads_end`.
    * *same_upper (same_lower)* the input is padded to match the output size. In case of odd padding value an extra padding is added at the end (at the beginning).
    * *valid* - do not use padding.
  * **Type**: string
  * **Default value**: None
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.

**Inputs**:

*   **1**: 4D or 5D input tensor. Required.

*   **2**: Convolution kernel tensor. Weights layout is GOIYX (GOIZYX for 3D convolution), which means that *X* is changing the fastest, then *Y*, then *Input*, *Output* and *Group*. The size of kernel and number of groups are derived from the shape of this input and aren't specified by any attribute. Required.


**Mathematical Formulation**

*   For the convolutional layer, the number of output features in each dimension is calculated using the formula:
\f[
n_{out} = \left ( \frac{n_{in} + 2p - k}{s} \right ) + 1
\f]
*   The receptive field in each layer is calculated using the formulas:
    *   Jump in the output feature map:
        \f[
        j_{out} = j_{in} * s
        \f]
    *   Size of the receptive field of output feature:
        \f[
        r_{out} = r_{in} + ( k - 1 ) * j_{in}
        \f]
    *   Center position of the receptive field of the first output feature:
        \f[
        start_{out} = start_{in} + ( \frac{k - 1}{2} - p ) * j_{in}
        \f]
    *   Output is calculated using the following formula:
        \f[
        out = \sum_{i = 0}^{n}w_{i}x_{i} + b
        \f]

**Example**

```xml
<layer type="GroupConvolution" ...>
    <data dilations="1,1" pads_begin="2,2" pads_end="2,2" strides="1,1"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>12</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
        <port id="1">
            <dim>4</dim>
            <dim>1</dim>
            <dim>3</dim>
            <dim>5</dim>
            <dim>5</dim>
        </port>
    </input>
    <output>
        <port id="2" precision="FP32">
            <dim>1</dim>
            <dim>4</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </output>
```
