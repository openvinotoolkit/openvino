GRN
===


.. meta::
  :description: Learn about GRN-1 - a normalization operation, which can be
                performed on a single input tensor.

**Versioned name**: *GRN-1*

**Category**: *Normalization*

**Short description**: *GRN* is the Global Response Normalization with L2 norm (across channels only).

**Detailed description**:

*GRN* computes the L2 norm across channels for input tensor with shape ``[N, C, ...]``. *GRN* does the following with the input tensor:

.. math::

   output[i0, i1, ..., iN] = x[i0, i1, ..., iN] / sqrt(sum[j = 0..C-1](x[i0, j, ..., iN]**2) + bias)


**Attributes**:

* *bias*

  * **Description**: *bias* is added to the sum of squares.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

**Inputs**

* **1**:  ``data`` - A tensor of type *T* and ``2 <= rank <= 4``. **Required.**

**Outputs**

* **1**: The result of *GRN* function applied to ``data`` input tensor. Normalized tensor of the same type and shape as the data input.

**Types**

* *T*: arbitrary supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="GRN">
       <data bias="1e-4"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>20</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
       </input>
       <output>
           <port id="0" precision="f32">
               <dim>1</dim>
               <dim>20</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
       </output>
   </layer>




