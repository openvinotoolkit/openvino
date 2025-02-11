SearchSorted
===============


.. meta::
  :description: Learn about SearchSorted - a sorting and maximization
                operation, which requires two input tensors.


**Versioned name**: *SearchSorted-15*

**Category**: *Sorting and maximization*

**Short description**: Determines the indices in the innermost dimension of a sorted sequence where elements should be inserted to maintain order.

**Detailed description**: *SearchSorted* operation determines the indices in the innermost dimension of a sorted sequence where elements should be inserted to maintain order. The operation is based on the binary search algorithm. The operation is performed on two input tensors: the first tensor contains a monotonically increasing sequence on the innermost dimension, and the second tensor contains the search values. The operation returns a tensor with the same shape as the second input tensor, containing the indices.

**Attributes**

* *right_mode*

  * **Description**: flag to control whether output would contain leftmost or rightmost indices for given values.
  * **Range of values**:

    * *true* - return the rightmost (last) suitable index for given value.
    * *false* - return the leftmost (first) suitable index for given value.
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

**Inputs**:

* **1**: ``sorted_sequence`` - ND input tensor of type *T* - cannot be a scalar, containing monotonically increasing sequence on the innermost dimension. **Required.**

* **2**: ``values`` - ND input tensor of type *T*, containing the search values. If sorted sequence is 1D, then the values can have any shape, otherwise the rank should be equal to the rank of sorted input. **Required.**

**Outputs**:

* **1**: Tensor of type *T_IND*, with the same shape as second input tensor ``values``, containing the indices.

**Types**

* *T*: any supported floating-point and integer type.

* *T_IND*: ``int64``.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="SearchSorted" ... >
       <data right_mode="true"/>
       <input>
           <port id="0">
               <dim>7</dim>
               <dim>256</dim>
               <dim>200</dim>
               <dim>200</dim>
           </port>
           <port id="1">
               <dim>7</dim>
               <dim>256</dim>
               <dim>200</dim>
               <dim>10</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="I64">
               <dim>7</dim>
               <dim>256</dim>
               <dim>200</dim>
               <dim>10</dim>
           </port>
       </output>
   </layer>
