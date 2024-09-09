Concat
======


.. meta::
  :description: Learn about Concat-1 - a data movement operation,
                which can be performed on arbitrary number of input tensors.

**Versioned name**: *Concat-1*

**Category**: *Data movement*

**Short description**: Concatenates arbitrary number of input tensors to a single output tensor along one axis.

**Attributes**:

* *axis*

  * **Description**: *axis* specifies dimension to concatenate along
  * **Range of values**: integer number. Negative value means counting dimension from the end. The range is ``[-R, R-1]``, where ``R`` is the rank of all inputs.
  * **Type**: int
  * **Required**: *yes*

**Inputs**:

* **1..N**: Arbitrary number of input tensors of type *T*. Types of all tensors should match. Rank of all tensors should match. The rank is positive, so scalars as inputs are not allowed. Shapes for all inputs should match at every position except ``axis`` position. At least one input is required.

**Outputs**:

* **1**: Tensor of the same type *T* as input tensor and shape ``[d1, d2, ..., d_axis, ...]``, where ``d_axis`` is a sum of sizes of input tensors along ``axis`` dimension.

**Types**

* *T*: any numeric type.

**Examples**

.. code-block:: xml
   :force:

   <layer id="1" type="Concat">
       <data axis="1" />
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>8</dim>  <!-- axis for concatenation -->
               <dim>50</dim>
               <dim>50</dim>
           </port>
           <port id="1">
               <dim>1</dim>
               <dim>16</dim>  <!-- axis for concatenation -->
               <dim>50</dim>
               <dim>50</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>32</dim>  <!-- axis for concatenation -->
               <dim>50</dim>
               <dim>50</dim>
           </port>
       </input>
       <output>
           <port id="0">
               <dim>1</dim>
               <dim>56</dim>  <!-- concatenated axis: 8 + 16 + 32 = 48 -->
               <dim>50</dim>
               <dim>50</dim>
           </port>
       </output>
   </layer>


.. code-block:: xml
   :force:

   <layer id="1" type="Concat">
       <data axis="-3" />
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>8</dim>  <!-- axis for concatenation -->
               <dim>50</dim>
               <dim>50</dim>
           </port>
           <port id="1">
               <dim>1</dim>
               <dim>16</dim>  <!-- axis for concatenation -->
               <dim>50</dim>
               <dim>50</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>32</dim>  <!-- axis for concatenation -->
               <dim>50</dim>
               <dim>50</dim>
           </port>
       </input>
       <output>
           <port id="0">
               <dim>1</dim>
               <dim>56</dim>  <!-- concatenated axis: 8 + 16 + 32 = 48 -->
               <dim>50</dim>
               <dim>50</dim>
           </port>
       </output>
   </layer>


