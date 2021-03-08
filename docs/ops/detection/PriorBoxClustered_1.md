## PriorBoxClustered <a name="PriorBoxClustered"></a> {#openvino_docs_ops_detection_PriorBoxClustered_1}

**Versioned name**: *PriorBoxClustered-1*

**Category**: Object detection

**Short description**: *PriorBoxClustered* operation generates prior boxes of specified sizes normalized to the input image size.

**Attributes**

* *width (height)*

  * **Description**: *width (height)* specifies desired boxes widths (heights) in pixels.
  * **Range of values**: floating point positive numbers
  * **Type**: float[]
  * **Default value**: 1.0
  * **Required**: *no*

* *clip*

  * **Description**: *clip* is a flag that denotes if each value in the output tensor should be clipped within [0,1].
  * **Range of values**:
    * false - clipping is not performed
    * true  - each value in the output tensor is within [0,1]
  * **Type**: boolean
  * **Default value**: true
  * **Required**: *no*

* *step (step_w, step_h)*

  * **Description**: *step (step_w, step_h)* is a distance between box centers. For example, *step* equal 85 means that the distance between neighborhood prior boxes centers is 85. If both *step_h* and *step_w* are 0 then they are updated with value of *step*. If after that they are still 0 then they are calculated as input image width(height) divided with first input width(height). 
  * **Range of values**: floating point positive number
  * **Type**: float
  * **Default value**: 0.0
  * **Required**: *no*

* *offset*

  * **Description**: *offset* is a shift of box respectively to top left corner. For example, *offset* equal 85 means that the shift of neighborhood prior boxes centers is 85.
  * **Range of values**: floating point positive number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

* *variance*

  * **Description**: *variance* denotes a variance of adjusting bounding boxes.
  * **Range of values**: floating point positive numbers
  * **Type**: float[]
  * **Default value**: []
  * **Required**: *no*

* *img_h (img_w)*

  * **Description**: *img_h (img_w)* specifies height (width) of input image. These attributes are taken from the second input `image_size` height(width) unless provided explicitly as the value for this attributes.
  * **Range of values**: floating point positive number
  * **Type**: float
  * **Default value**: 0
  * **Required**: *no*

**Inputs**:

*   **1**: `output_size` - 1D tensor with two integer elements `[height, width]`. Specifies the spatial size of generated grid with boxes. Required.

*   **2**: `image_size` - 1D tensor with two integer elements `[image_height, image_width]` that specifies shape of the image for which boxes are generated. Optional.

**Outputs**:

*   **1**: 2D tensor of shape `[2, 4 * height * width * priors_per_point]` with box coordinates. The `priors_per_point` is the number of boxes generated per each grid element. The number depends on layer attribute values.

**Detailed description**

*PriorBoxClustered* computes coordinates of prior boxes by following:
1.  Calculates the *center_x* and *center_y* of prior box:
    \f[
    W \equiv Width \quad Of \quad Image
    \f]
    \f[
    H \equiv Height \quad Of \quad Image
    \f]
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
2.  For each \f$s \subset \left( 0, W \right )\f$ calculates the prior boxes coordinates:
    \f[
    xmin = \frac{center_x - \frac{width_s}{2}}{W}
    \f]
    \f[
    ymin = \frac{center_y - \frac{height_s}{2}}{H}
    \f]
    \f[
    xmax = \frac{center_x - \frac{width_s}{2}}{W}
    \f]
    \f[
    ymax = \frac{center_y - \frac{height_s}{2}}{H}
    \f]
If *clip* is defined, the coordinates of prior boxes are recalculated with the formula:
\f$coordinate = \min(\max(coordinate,0), 1)\f$

**Example**

```xml
<layer type="PriorBoxClustered" ... >
    <data clip="false" height="44.0,10.0,30.0,19.0,94.0,32.0,61.0,53.0,17.0" offset="0.5" step="16.0" variance="0.1,0.1,0.2,0.2" width="86.0,13.0,57.0,39.0,68.0,34.0,142.0,50.0,23.0"/>
    <input>
        <port id="0">
            <dim>2</dim>        <!-- [10, 19] -->
        </port>
        <port id="1">
            <dim>2</dim>        <!-- [180, 320] -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>2</dim>
            <dim>6840</dim>
        </port>
    </output>
</layer>
```