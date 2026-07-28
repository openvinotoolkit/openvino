DeformablePSROIPooling
======================


.. meta::
  :description: Learn about DeformablePSROIPooling-1 - an object detection operation, which
                can be performed on two or three input tensors in OpenVINO.

**Versioned name**: *DeformablePSROIPooling-1*

**Category**: *Object detection*

**Short description**: *DeformablePSROIPooling* computes deformable position-sensitive pooling of regions of interest specified by input.

**Detailed description**: `Reference <https://arxiv.org/abs/1703.06211>`__.

*DeformablePSROIPooling* operation takes two or three input tensors: with position score maps, with regions of interests (ROI, box coordinates) and an optional tensor with transformation values (normalized offsets for ROI bins coordinates).
If only two inputs are provided, position sensitive pooling with regular ROI bins position is calculated (non-deformable).
If third input is provided, each bin position is transformed by adding corresponding offset to the bin left top corner coordinates. Third input values are usually calculated by regular position sensitive pooling layer, so non-deformable mode (DeformablePSROIPooling with two inputs).
The ROI coordinates are specified as five element tuples: ``[batch_id, x_1, y_1, x_2, y_2]`` in absolute values.

This operation is compatible with `Apache MXNet DeformablePSROIPooling <https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/contrib/symbol/index.html#mxnet.contrib.symbol.DeformablePSROIPooling>`__ cases where ``group_size`` is equal to ``pooled_size``.

**Attributes**

* *output_dim*

  * **Description**: *output_dim* is the number of the output channels, size of output `C` dimension.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *spatial_scale*

  * **Description**: *spatial_scale* is a multiplicative spatial scale factor to translate ROI coordinates from their input original size to the pooling input. Ratio of the input score map size to the original image size.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *group_size*

  * **Description**: *group_size* is the number of horizontal bins per row to divide single ROI area. Total number of bins can be calculated as ``group_size*group_size``. It defines pooled width and height, so output ``H_out`` and ``W_out`` dimensions (always equal). Square of the ``group_size`` is also the number to divide input channels ``C_in`` dimension and split it into ``C_in \\ group_size*group_size`` groups. Each group corresponds to the exactly one output channel and ROI's bins are spread over input channel group members.

  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

* *mode*

  * **Description**: *mode* specifies mode for pooling.
  * **Range of values**:

    * *bilinear_deformable* - perform pooling with bilinear interpolation over single ROI bin. For each ROI bin average of his interpolated ``spatial_bins_x*spatial_bins_y`` sub-bins values is calculated.
  * **Type**: ``string``
  * **Default value**: *bilinear_deformable*
  * **Required**: *no*

* *spatial_bins_x*

  * **Description**: *spatial_bins_x* specifies number of horizontal sub-bins (bilinear interpolation samples) to divide ROI single bin.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

* *spatial_bins_y*

  * **Description**: *spatial_bins_y* specifies number of vertical sub-bins (bilinear interpolation samples) to divide ROI single bin.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

* *trans_std*

  * **Description**: *trans_std* is the value that all third input values (offests) are multiplied with to modulate the magnitude of the offsets.
  * **Range of values**: floating-point number
  * **Type**: ``float``
  * **Default value**: 1
  * **Required**: *no*

* *part_size*

  * **Description**: *part_size* is the size of ``H`` and ``W`` dimensions of the third input (offsets). Basically it is the height and width of the third input with transformation values.
  * **Range of values**: positive integer number
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

* **1**: 4D input tensor of type *T* and shape ``[N_in, C_in, H_in, W_in]`` with position sensitive score maps. **Required.**
* **2**: 2D input tensor of type *T* and shape ``[NUM_ROIS, 5]``. It contains a list of five element tuples describing a single ROI (region of interest): ``[batch_id, x_1, y_1, x_2, y_2]``. **Required.** Batch indices must be in the range of ``[0, N_in-1]``.
* **3**: 4D input tensor of type *T* and shape ``[NUM_ROIS, 2*NUM_CLASSES, group_size, group_size]`` with transformation values. It contains normalized ``[0, 1]`` offsets for each ROI bin left top corner coordinates. Channel dimension is multiplied by ``2`` because of encoding two ``(x, y)`` coordinates. **Optional.**

**Outputs**:

*   **1**: 4D output tensor of type *T* shape ``[NUM_ROIS, output_dim, group_size, group_size]`` with ROIs score maps.

**Types**:

* *T*: Any floating-point type.

**Example**

* Two inputs (without offsets)

.. code-block:: xml
   :force:

   <layer ... type="DeformablePSROIPooling" ... >
       <data spatial_scale="0.0625" output_dim="882" group_size="3" mode="bilinear_deformable" spatial_bins_x="4" spatial_bins_y="4" trans_std="0.0" part_size="3"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>7938</dim>
               <dim>63</dim>
               <dim>38</dim>
           </port>
           <port id="1">
               <dim>300</dim>
               <dim>5</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="FP32">
               <dim>300</dim>
               <dim>882</dim>
               <dim>3</dim>
               <dim>3</dim>
           </port>
       </output>
   </layer>


* Three inputs (with offsets)

.. code-block:: xml
   :force:

   <layer ... type="DeformablePSROIPooling" ... >
       <data group_size="7" mode="bilinear_deformable" output_dim="8" part_size="7" spatial_bins_x="4" spatial_bins_y="4" spatial_scale="0.0625" trans_std="0.1"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>392</dim>
               <dim>38</dim>
               <dim>63</dim>
           </port>
           <port id="1">
               <dim>300</dim>
               <dim>5</dim>
           </port>
           <port id="2">
               <dim>300</dim>
               <dim>2</dim>
               <dim>7</dim>
               <dim>7</dim>
           </port>
       </input>
       <output>
           <port id="3" precision="FP32">
               <dim>300</dim>
               <dim>8</dim>
               <dim>7</dim>
               <dim>7</dim>
           </port>
       </output>
   </layer>


