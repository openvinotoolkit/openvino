## DeformableConvolution<a name="DeformableConvolution"></a> {#openvino_docs_ops_convolution_DeformableConvolution_1}

**Versioned name**: *DeformableConvolution-1*

**Category**: Convolution

**Short description**: Computes 1D, 2D or 3D deformable convolution of input and kernel tensors.

**Detailed description**: *Deformable Convolution* is similar to regular *Convolution* but its receptive field is deformed because of additional spatial offsets used during input sampling. More thorough explanation can be found in [Deformable Convolutions Demystified](https://towardsdatascience.com/deformable-convolutions-demystified-2a77498699e8) and [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211).

**Attributes**:

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the `(z, y, x)` axes for 3D convolutions and `(y, x)` axes for 2D convolutions. For example, *strides* equal `4,2,1` means sliding the filter 4 pixel at a time over depth dimension, 2 over height dimension and 1 over width dimension.
  * **Range of values**: integer values starting from `0`
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal `1,2` means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: integer values starting from `0`
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal `1,2` means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: integer values starting from `0`
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* denotes the distance in width and height between elements (weights) in the filter. For example, *dilation* equal `1,1` means that all the elements in the filter are neighbors, so it is the same as for the usual convolution. *dilation* equal `2,2` means that all the elements in the filter are matched not to adjacent elements in the input matrix, but to those that are adjacent with distance 1.
  * **Range of values**: integer value starting from `0`
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:
    * *explicit* - use explicit padding values from *pads_begin* and *pads_end*.
    * *same_upper* - the input is padded to match the output size. In case of odd padding value an extra padding is added at the end.
    * *same_lower* - the input is padded to match the output size. In case of odd padding value an extra padding is added at the beginning.
    * *valid* - do not use padding.
  * **Type**: `string`
  * **Default value**: explicit
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.


* *group*

  * **Description**: *group* is the number of groups which *output* and *input* should be split into. For example, *group* equal to 1 means that all filters are applied to the whole input (usual convolution), *group* equal to 2 means that both *input* and *output* channels are separated into two groups and the *i-th output* group is connected to the *i-th input* group channel. *group* equal to a number of output feature maps implies depth-wise separable convolution.
  * **Range of values**: integer value starting from `1`
  * **Type**: `int`
  * **Default value**: `1`
  * **Required**: *no*

* *deformable_group*

  * **Description**: *deformable_group* is the number of groups which deformable values and *output* should be split into along the channel axis. Apply the deformable convolution using the i-th part of the offset part on the i-th out.
  * **Range of values**: integer value starting from `1`
  * **Type**: `int`
  * **Default value**: `1`
  * **Required**: *no*

**Inputs**:

*   **1**: Input tensor of type *T* and rank 3, 4 or 5. Layout is `NCZYX` (number of batches, number of channels, spatial axes Z, Y, X). Required.

*   **2**: Offsets tensor of type *T* and rank 3, 4 or 5. Layout is `NCZYX*`(number of batches, number of channels, spatial axes Z, Y, X). Required.

*   **3**: Kernel tensor of type *T* and rank 3, 4 or 5. Layout is `OIZYX` (number of output channels, number of input channels, spatial axes Z, Y, X). Required.
*   **Note**: Type of the convolution (1D, 2D or 3D) is derived from the rank of the input tensors and not specified by any attribute:
      * 1D convolution (input tensors rank 3) means that there is only one spatial axis X
      * 2D convolution (input tensors rank 4) means that there are two spatial axes Y, X
      * 3D convolution (input tensors rank 5) means that there are three spatial axes Z, Y, X


**Outputs**:

*   **1**: Output tensor of type *T* and rank 3, 4 or 5. Layout is NOZYX (number of batches, number of kernel output channels, spatial axes Z, Y, X).

**Types**:

* *T*: Any floating point type.
 
**Example**

1D Convolution
```xml
<layer type="DeformableConvolution" ...>
    <data dilations="1" pads_begin="0" pads_end="0" strides="2" auto_pad="valid" group="1" deformable_group="1"/>
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
<layer type="DeformableConvolution" ...>
    <data dilations="1,1" pads_begin="2,2" pads_end="2,2" strides="1,1" auto_pad="explicit"  group="1" deformable_group="1"/>
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
<layer type="DeformableConvolution" ...>
    <data dilations="2,2,2" pads_begin="0,0,0" pads_end="0,0,0" strides="3,3,3" auto_pad="explicit" group="1" deformable_group="1"/>
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