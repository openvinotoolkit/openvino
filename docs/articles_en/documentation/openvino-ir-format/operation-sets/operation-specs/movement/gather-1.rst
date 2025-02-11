Gather
======


.. meta::
  :description: Learn about Gather-1 - a data movement operation,
                which can be performed on three required input tensors.

**Versioned name:** *Gather-1*

**Category:** *Data movement*

**Short description:** *Gather* operation takes slices of data in the first input tensor according
to the indices specified in the second input tensor and axis from the third input.

**Detailed description**

.. code-block:: sh

    output[p_0, p_1, ..., p_{axis-1}, i, ..., j, ...] =
      input1[p_0, p_1, ..., p_{axis-1}, input2[i, ..., j], ...]

Where ``axis`` is the value from the third input.

**Attributes**: *Gather* has no attributes

**Inputs**

* **1**:  Tensor with arbitrary data. **Required.**
* **2**:  Tensor with indices to gather. The values for indices are in the range ``[0, input1[axis] - 1]``. **Required.**
* **3**:  Scalar or 1D tensor *axis* is a dimension index to gather data from. For example, *axis* equal
  to 1 means that gathering is performed over the first dimension. Negative value means reverse indexing.
  Allowed values are from ``[-len(input1.shape), len(input1.shape) - 1]``. **Required.**

**Outputs**

* **1**: The resulting tensor that consists of elements from the first input tensor gathered by indices
  from the second input tensor. Shape of the tensor is ``[input1.shape[:axis], input2.shape, input1.shape[axis + 1:]]``

**Example**

.. code-block:: xml
   :force:

   <layer id="1" type="Gather" ...>
       <input>
           <port id="0">
               <dim>6</dim>
               <dim>12</dim>
               <dim>10</dim>
               <dim>24</dim>
           </port>
           <port id="1">
               <dim>15</dim>
               <dim>4</dim>
               <dim>20</dim>
               <dim>28</dim>
           </port>
           <port id="2"/>        <!--  axis = 1  -->
       </input>
       <output>
           <port id="2">
               <dim>6</dim>      <!-- embedded dimension from the 1st input -->
               <dim>15</dim>     <!-- embedded dimension from the 2nd input -->
               <dim>4</dim>      <!-- embedded dimension from the 2nd input -->
               <dim>20</dim>     <!-- embedded dimension from the 2nd input -->
               <dim>28</dim>     <!-- embedded dimension from the 2nd input -->
               <dim>10</dim>     <!-- embedded dimension from the 1st input -->
               <dim>24</dim>     <!-- embedded dimension from the 1st input -->
           </port>
       </output>
   </layer>




