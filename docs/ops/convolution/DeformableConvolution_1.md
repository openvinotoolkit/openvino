## DeformableConvolution<a name="DeformableConvolution"></a> {#openvino_docs_ops_convolution_DeformableConvolution_1}

**Versioned name**: *DeformableConvolution-1*

**Category**: Convolution

**Detailed description**: [Reference](https://arxiv.org/abs/1703.06211)

**Attributes**

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the (z, y, x) axes for 3D convolutions and (y, x) axes for 2D convolutions. For example, *strides* equal *4,2,1* means sliding the filter 4 pixel at a time over depth dimension, 2 over height dimension and 1 over width dimension.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal *1,2* means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal *1,2* means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* denotes the distance in width and height between elements (weights) in the filter. For example, *dilation* equal *1,1* means that all the elements in the filter are neighbors, so it is the same as for the usual convolution. *dilation* equal *2,2* means that all the elements in the filter are matched not to adjacent elements in the input matrix, but to those that are adjacent with distance 1.
  * **Range of values**: integer value starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:
    * None (not specified): use explicit padding values.
    * *same_upper (same_lower)* the input is padded to match the output size. In case of odd padding value an extra padding is added at the end (at the beginning).
    * *valid* - do not use padding.
  * **Type**: string
  * **Default value**: None
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.

* *group*

  * **Description**: *group* is the number of groups which *output* and *input* should be split into. For example, *group* equal to 1 means that all filters are applied to the whole input (usual convolution), *group* equal to 2 means that both *input* and *output* channels are separated into two groups and the *i-th output* group is connected to the *i-th input* group channel. *group* equal to a number of output feature maps implies depth-wise separable convolution.
  * **Range of values**: integer value starting from 1
  * **Type**: int
  * **Default value**: 1
  * **Required**: *no*

* *deformable_group*

  * **Description**: *deformable_group* is the number of groups which deformable values and *output* should be split into along the channel axis. Apply the deformable convolution using the i-th part of the offset part on the i-th out.
  * **Range of values**: integer value starting from 1
  * **Type**: int
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

*   **1**: Input tensor of rank 3 or greater. Required.

*   **2**: Deformable values tensor of rank 3 or higher. Required.

*   **3**: Convolution kernel tensor. Weights layout is OIYX (OIZYX for 3D convolution), which means that *X* is changing the fastest, then *Y*, then *Input* then *Output*. The size of kernel is derived from the shape of this input and not specified by any attribute. Required.

**Example**

```xml
<layer ... type="DeformableConvolution" ... >
        <data dilations="1,1" pads_begin="2,2" pads_end="3,3" strides="2,2"/>
        <input> ... </input>
        <output> ... </output>
</layer>
```