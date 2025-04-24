AdaptiveMaxPool
===============


.. meta::
  :description: Learn about AdaptiveMaxPool-8 - a pooling operation, which can
                be performed on two required input tensors.

**Versioned name**: *AdaptiveMaxPool-8*

**Category**: *Pooling*

**Short description**: Applies max pooling with adaptive kernel size over the input.

**Detailed description**: This operation calculates the output based on the first input and ``output_size`` determined by the second input.
The kernel dimensions are calculated using the following formulae for the ``NCDHW`` input case:

.. math::

  \begin{array}{lcl}
  d_{start} &=& \lfloor i \cdot \frac{D_{in}}{D_{out}}\rfloor\\
  d_{end}   &=& \lceil(i+1) \cdot \frac{D_{in}}{D_{out}}\rceil\\
  h_{start} &=& \lfloor j \cdot \frac{H_{in}}{H_{out}}\rfloor\\
  h_{end}   &=& \lceil(j+1) \cdot \frac{H_{in}}{H_{out}}\rceil\\
  w_{start} &=& \lfloor k \cdot \frac{W_{in}}{W_{out}}\rfloor\\
  w_{end}   &=& \lceil(k+1) \cdot \frac{W_{in}}{W_{out}}\rceil
  \end{array}

The output is calculated following this formula:

.. math::

   Output(i,j,k) = max(Input[d_{start}:d_{end}, h_{start}:h_{end}, w_{start}:w_{end}])

**Attributes**:

* *index_element_type*

  * **Description**: the type of the second output containing indices
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i64"
  * **Required**: *no*

**Inputs**:

* **1**: 3D, 4D, or 5D input tensor of shape ``[N, C, H]``, ``[N, C, H, W]`` or ``[N, C, D, H, W]`` and type *T*. **Required.**
* **2**: 1D tensor describing output shape for spatial dimensions. Can be ``[H_out]`` for 3D input, ``[H_out, W_out]`` for 4D input, ``[D_out, H_out, W_out]`` for 5D input and of type *T_SHAPE*. **Required.**

**Outputs**:

* **1**: Output of type *T* and shape ``[N, C, H_out]``, ``[N, C, H_out, W_out]`` or ``[N, C, D_out, H_out, W_out]``.
* **2**: Output of type specified by *index_element_type* and same shape as the first output containing indices of elements in the first output. The values of indices are computed as if input spatial dimensions were flatten, so the values are in the range ``[0, H * W * D)``.

**Types**

* *T*: floating-point type.
* *T_SHAPE*: ``int32`` or ``int64``.

**Examples**

.. code-block:: xml
   :force:

   <layer ... type="AdaptiveMaxPool" ... >
       <data output_type="i64"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>3</dim>
               <dim>32</dim>
               <dim>32</dim>
           </port>
       </input>
       <input>
           <port id="1">
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>1</dim>
               <dim>3</dim>
               <dim>16</dim>
               <dim>16</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>3</dim>
               <dim>16</dim>
               <dim>16</dim>
           </port>
       </output>
   </layer>


