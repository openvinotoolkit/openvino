PSROIPooling
============


.. meta::
  :description: Learn about PSROIPooling-1 - an object detection operation,
                which can be performed on two required input tensors.

**Versioned name**: *PSROIPooling-1*

**Category**: *Object detection*

**Short description**: *PSROIPooling* computes position-sensitive pooling on regions of interest specified by input.

**Detailed description**: `Reference <https://arxiv.org/pdf/1703.06211.pdf>`__.

*PSROIPooling* operation takes two input blobs: with feature maps and with regions of interests (box coordinates).
The latter is specified as five element tuples: *[batch_id, x_1, y_1, x_2, y_2]*.
ROIs coordinates are specified in absolute values for the average mode and in normalized values (to *[0,1]* interval) for bilinear interpolation.

**Attributes**

* *output_dim*

  * **Description**: *output_dim* is a pooled output channel number.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *group_size*

  * **Description**: *group_size* is the number of groups to encode position-sensitive score maps.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

* *spatial_scale*

  * **Description**: *spatial_scale* is a multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *mode*

  * **Description**: *mode* specifies mode for pooling.
  * **Range of values**:

    * *average* - perform average pooling
    * *bilinear* - perform pooling with bilinear interpolation
  * **Type**: string
  * **Default value**: *average*
  * **Required**: *no*

* *spatial_bins_x*

  * **Description**: *spatial_bins_x* specifies numbers of bins to divide the input feature maps over width. Used for "bilinear" mode only.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

* *spatial_bins_y*

  * **Description**: *spatial_bins_y* specifies numbers of bins to divide the input feature maps over height.  Used for "bilinear" mode only.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

* **1**: 4D input tensor with shape ``[N, C, H, W]`` and type *T*  with feature maps. **Required.**

* **2**: 2D input tensor with shape ``[num_boxes, 5]``. It contains a list of five element tuples that describe a region of interest: ``[batch_id, x_1, y_1, x_2, y_2]``. **Required.**
  Batch indices must be in the range of ``[0, N-1]``.

**Outputs**:

*   **1**: 4D output tensor with areas copied and interpolated from the 1st input tensor by coordinates of boxes from the 2nd input.

**Types**

* *T*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="PSROIPooling" ... >
       <data group_size="6" mode="bilinear" output_dim="360" spatial_bins_x="3" spatial_bins_y="3" spatial_scale="1"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>3240</dim>
               <dim>38</dim>
               <dim>38</dim>
           </port>
           <port id="1">
               <dim>100</dim>
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>100</dim>
               <dim>360</dim>
               <dim>6</dim>
               <dim>6</dim>
           </port>
       </output>
   </layer>




