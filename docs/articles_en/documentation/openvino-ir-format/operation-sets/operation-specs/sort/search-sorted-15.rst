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

* *right*

  * **Description**: If False, set the first suitable index. If True, return the last suitable index for given value. Default is False.
  * **Range of values**: true or false
  * **Type**: boolean
  * **Default value**: false
  * **Required**: *no*

**Inputs**:

* **1**: ``sorted`` - ND input tensor of type *T* - cannot be a scalar, containing monotonically increasing sequence on the innermost dimension. **Required.**

* **2**: ``values`` - ND input tensor of type *T*, containing the search values. If sorted sequence is 1D, then the values can have any shape, otherwise the rank should be equal to the rank of sorted input. **Required.**

**Outputs**:

* **1**: Tensor of type *TOut*, with the same shape as second input tensor, containing the indices.

**Types**

* *T*: any supported floating-point and integer type.

* *TOut*: int64.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="SearchSorted" ... >
       <data right="True"/>
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
           <port id="0" precision="I64">
               <dim>7</dim>
               <dim>256</dim>
               <dim>200</dim>
               <dim>10</dim>
           </port>
       </output>
   </layer>
