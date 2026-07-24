BevPoolV2
=========


.. meta::
  :description: Learn about BevPoolV2-1 - a pooling operation for Bird's-Eye-View
                (BEV) feature aggregation from multi-camera inputs.

**Versioned name**: *BevPoolV2-1*

**Category**: *Pooling*

**Short description**: Aggregates multi-camera image features into a Bird's-Eye-View (BEV) feature map by performing depth-weighted scatter-accumulate pooling.

**Detailed description**:

BevPoolV2 implements the pooling kernel from the BEVPoolV2 algorithm used in BEV-based 3D perception pipelines (e.g., BEVDet, BEVDepth). It takes per-camera feature maps (*cf*), per-point depth weights (*dw*), a flat index array (*idx*) that maps each BEV voxel contribution to a position in *dw*, and an interval table (*itv*) that groups those contributions by BEV output cell.

For each interval (BEV cell), the operation accumulates the weighted feature sum over all contributing depth points:

.. math::

   \text{out}[b, c, h, w] = \sum_{i=\text{start}}^{\text{end}-1}
       cf\!\left[\text{cam}(idx[i]),\, \text{feat}(idx[i]),\, c\right]
       \cdot dw\!\left[idx[i]\right]

where :math:`\text{cam}(\cdot)` and :math:`\text{feat}(\cdot)` derive the camera index and spatial feature index from the flat *dw* offset using:

.. math::

   \text{cam}(k) &= \left\lfloor k \;/\; (D \cdot H_{img} \cdot W_{img}) \right\rfloor \\
   \text{feat}(k) &= k \;\bmod\; (H_{img} \cdot W_{img})

and :math:`D = \lfloor (d\_bound_{max} - d\_bound_{min}) / d\_bound_{step} \rfloor` is the number of depth bins.

The output layout is ``[N, C_{out}, H_{feat}, W_{feat}]`` (NCHW).

**Attributes**:

* *input_channels*

  * **Description**: number of input feature channels in *cf* (dimension index 3).
  * **Range of values**: positive integer
  * **Type**: ``uint32``
  * **Required**: *yes*

* *output_channels*

  * **Description**: number of output feature channels; equals the size of the channel dimension in the output.
  * **Range of values**: positive integer
  * **Type**: ``uint32``
  * **Required**: *yes*

* *image_width*

  * **Description**: spatial width of the camera feature map (dimension index 2 of *cf*).
  * **Range of values**: positive integer
  * **Type**: ``uint32``
  * **Required**: *yes*

* *image_height*

  * **Description**: spatial height of the camera feature map (dimension index 1 of *cf*).
  * **Range of values**: positive integer
  * **Type**: ``uint32``
  * **Required**: *yes*

* *feature_width*

  * **Description**: width of the output BEV feature map.
  * **Range of values**: positive integer
  * **Type**: ``uint32``
  * **Required**: *yes*

* *feature_height*

  * **Description**: height of the output BEV feature map.
  * **Range of values**: positive integer
  * **Type**: ``uint32``
  * **Required**: *yes*

* *x_bound_min*, *x_bound_max*, *x_bound_step*

  * **Description**: X-axis range and resolution of the BEV grid ``[min, max, step]``. ``step`` must be positive; ``max >= min``.
  * **Type**: ``float``
  * **Required**: *yes*

* *y_bound_min*, *y_bound_max*, *y_bound_step*

  * **Description**: Y-axis range and resolution of the BEV grid ``[min, max, step]``. ``step`` must be positive; ``max >= min``.
  * **Type**: ``float``
  * **Required**: *yes*

* *z_bound_min*, *z_bound_max*, *z_bound_step*

  * **Description**: Z-axis range and resolution of the BEV grid ``[min, max, step]``. ``step`` must be positive; ``max >= min``.
  * **Type**: ``float``
  * **Required**: *yes*

* *d_bound_min*, *d_bound_max*, *d_bound_step*

  * **Description**: Depth-axis range and resolution used to compute the number of depth bins :math:`D`. ``step`` must be positive; ``max >= min``.
  * **Type**: ``float``
  * **Required**: *yes*

**Inputs**:

* **1**: ``cf`` — 4D tensor of shape ``[N, H_img, W_img, C_in]`` and type *T_FP*. Per-camera image feature map. **Required.**
* **2**: ``dw`` — tensor of type *T_FP*. Flat array of per-point depth weights; length equals the total number of depth-voxel contributions across all cameras. **Required.**
* **3**: ``idx`` — 1D tensor of type *T_INT*. Flat index array mapping each contribution slot to a position in *dw*. **Required.**
* **4**: ``itv`` — 1D tensor of shape ``[3 * K]`` or 2D tensor of shape ``[K, 3]`` and type *T_INT*. Interval table with *K* BEV cells; each row is ``[start, end, bev_offset]`` where ``start`` and ``end`` are half-open bounds into *idx*, and ``bev_offset`` is the flat write index into the output tensor. **Required.**

**Outputs**:

* **1**: 4D output tensor of shape ``[N, C_out, H_feat, W_feat]`` and type *T_FP*. BEV feature map with accumulated depth-weighted features.

**Types**

* *T_FP*: ``f16`` or ``f32``. All floating-point inputs must share the same element type.
* *T_INT*: ``i32`` or ``i64``.

**Examples**

.. code-block:: xml
   :force:

   <layer ... type="BevPoolV2" ... >
       <data input_channels="64" output_channels="64"
             image_width="16" image_height="16"
             feature_width="128" feature_height="128"
             x_bound_min="-51.2" x_bound_max="51.2" x_bound_step="0.8"
             y_bound_min="-51.2" y_bound_max="51.2" y_bound_step="0.8"
             z_bound_min="-5.0"  z_bound_max="3.0"  z_bound_step="8.0"
             d_bound_min="1.0"   d_bound_max="60.0" d_bound_step="0.5"/>
       <input>
           <port id="0"><!-- cf: [N, H_img, W_img, C_in] -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>16</dim>
               <dim>64</dim>
           </port>
           <port id="1"><!-- dw: flat depth weights -->
               <dim>131072</dim>
           </port>
           <port id="2"><!-- idx: flat indices -->
               <dim>131072</dim>
           </port>
           <port id="3"><!-- itv: [K, 3] interval table -->
               <dim>16384</dim>
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="4"><!-- output BEV: [N, C_out, H_feat, W_feat] -->
               <dim>1</dim>
               <dim>64</dim>
               <dim>128</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>
