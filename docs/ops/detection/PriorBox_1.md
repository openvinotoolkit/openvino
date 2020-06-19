## PriorBox<a name="PriorBox"></a>

**Versioned name**: *PriorBox-1*

**Category**: Object detection

**Short description**: *PriorBox* operation generates prior boxes of specified sizes and aspect ratios across all dimensions.

**Attributes**:

* *min_size (max_size)*

  * **Description**: *min_size (max_size)* is the minimum (maximum) box size (in pixels). For example, *min_size (max_size)* equal 15 means that the minimum (maximum) box size is 15.
  * **Range of values**: positive floating point numbers
  * **Type**: float[]
  * **Default value**: []
  * **Required**: *no*

* *aspect_ratio*

  * **Description**: *aspect_ratio* is a variance of aspect ratios. Duplicate values are ignored. For example, *aspect_ratio* equal "2.0,3.0" means that for the first box aspect_ratio is equal to 2.0 and for the second box is 3.0.
  * **Range of values**: set of positive integer numbers
  * **Type**: float[]
  * **Default value**: []
  * **Required**: *no*

* *flip*

  * **Description**: *flip* is a flag that denotes that each *aspect_ratio* is duplicated and flipped. For example, *flip* equals 1 and *aspect_ratio* equals to "4.0,2.0" mean that aspect_ratio is equal to "4.0,2.0,0.25,0.5".
  * **Range of values**:
    * False - each *aspect_ratio* is flipped
    * True  - each *aspect_ratio* is not flipped
  * **Type**: boolean
  * **Default value**: False
  * **Required**: *no*

* *clip*

  * **Description**: *clip* is a flag that denotes if each value in the output tensor should be clipped to [0,1] interval.
  * **Range of values**:
    * False - clipping is not performed
    * True - each value in the output tensor is clipped to [0,1] interval.
  * **Type**: boolean
  * **Default value**: False
  * **Required**: *no*

* *step*

  * **Description**: *step* is a distance between box centers. For example, *step* equal 85 means that the distance between neighborhood prior boxes centers is 85.
  * **Range of values**: floating point non-negative number
  * **Type**: float
  * **Default value**: 0
  * **Required**: *no*

* *offset*

  * **Description**: *offset* is a shift of box respectively to top left corner. For example, *offset* equal 85 means that the shift of neighborhood prior boxes centers is 85.
  * **Range of values**: floating point non-negative number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

* *variance*

  * **Description**: *variance* denotes a variance of adjusting bounding boxes. The attribute could contain 0, 1 or 4 elements.
  * **Range of values**: floating point positive numbers
  * **Type**: float[]
  * **Default value**: []
  * **Required**: *no*

* *scale_all_sizes*

  * **Description**: *scale_all_sizes* is a flag that denotes type of inference. For example, *scale_all_sizes* equals 0 means that the PriorBox layer is inferred in MXNet-like manner. In particular, *max_size* attribute is ignored.
  * **Range of values**:
    * False - *max_size* is ignored
    * True  - *max_size* is used
  * **Type**: boolean
  * **Default value**: True
  * **Required**: *no*

* *fixed_ratio*

    * **Description**: *fixed_ratio* is an aspect ratio of a box. For example, *fixed_ratio* equal to 2.000000 means that the aspect ratio for the first box aspect ratio is 2.
    * **Range of values**: a list of positive floating-point numbers
    * **Type**: `float[]`
    * **Default value**: None
    * **Required**: *no*

* *fixed_size*

    * **Description**: *fixed_size* is an initial box size (in pixels). For example, *fixed_size* equal to 15 means that the initial box size is 15.
    * **Range of values**: a list of positive floating-point numbers
    * **Type**: `float[]`
    * **Default value**: None
    * **Required**: *no*

* *density*

    * **Description**: *density* is the square root of the number of boxes of each type. For example, *density* equal to 2 means that the first box generates four boxes of the same size and with the same shifted centers.
    * **Range of values**: a list of positive floating-point numbers
    * **Type**: `float[]`
    * **Default value**: None
    * **Required**: *no*

**Inputs**:

*   **1**: `output_size` - 1D tensor with two integer elements `[height, width]`. Specifies the spatial size of generated grid with boxes. Required.

*   **2**: `image_size` - 1D tensor with two integer elements `[image_height, image_width]` that specifies shape of the image for which boxes are generated. Required.

**Outputs**:

*   **1**: 2D tensor of shape `[2, 4 * height * width * priors_per_point]` with box coordinates. The `priors_per_point` is the number of boxes generated per each grid element. The number depends on layer attribute values.

**Detailed description**:

*PriorBox* computes coordinates of prior boxes by following:
1.  First calculates *center_x* and *center_y* of prior box:
    \f[
    W \equiv Width \quad Of \quad Image
    \f]
    \f[
    H \equiv Height \quad Of \quad Image
    \f]
    *   If step equals 0:
        \f[
        center_x=(w+0.5)
        \f]
        \f[
        center_y=(h+0.5)
        \f]
    *   else:
        \f[
        center_x=(w+offset)*step
        \f]
        \f[
        center_y=(h+offset)*step
        \f]
        \f[
        w \subset \left( 0, W \right )
        \f]
        \f[
        h \subset \left( 0, H \right )
        \f]
2.  Then, for each \f$ s \subset \left( 0, min_sizes \right ) \f$ calculates coordinates of prior boxes:
    \f[
    xmin = \frac{\frac{center_x - s}{2}}{W}
    \f]
    \f[
    ymin = \frac{\frac{center_y - s}{2}}{H}
    \f]
    \f[
    xmax = \frac{\frac{center_x + s}{2}}{W}
    \f]
    \f[
    ymin = \frac{\frac{center_y + s}{2}}{H}
    \f]

**Example**

```xml
<layer type="PriorBox" ...>
    <data aspect_ratio="2.0" clip="0" density="" fixed_ratio="" fixed_size="" flip="1" max_size="38.46" min_size="16.0" offset="0.5" step="16.0" variance="0.1,0.1,0.2,0.2"/>
    <input>
        <port id="0">
            <dim>2</dim>        <!-- values: [24, 42] -->
        </port>
        <port id="1">
            <dim>2</dim>        <!-- values: [384, 672] -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>2</dim>
            <dim>16128</dim>
        </port>
    </output>
</layer>
```