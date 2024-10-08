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
  * **Description**: *right*  if False, return the first suitable location that is found. If True, return the last such index. If no suitable index is found, return 0 for non-numerical value (e.g. nan, inf) or the size of innermost dimension within sorted_sequence (one pass the last index of the innermost dimension). In other words, if False, get the lower bound index for each value in values on the corresponding innermost dimension of the sorted_sequence. If True, get the upper bound index instead. Default value is False. Side does the same and is preferred. It will error if side is set to “left” while this is True.
  * **Range of values**: true or false
  * **Type**: boolean
  * **Default value**: false
  * **Required**: *no*

**Inputs**:

* **1**: ND input tensor of type *T*, containing monotonically increasing sequence on the innermost dimension **Required.**

* **2**: ND input tensor of type *T*, containing the search values. **Required.**

**Outputs**:

* **1**: Tensor of type *TOut*, with the same shape as second input tensor, containing the indices.

**Types**

* *T*: any supported floating-point and integer type.

* *TOut*: int64.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="SearchSorted" ... >
       <right="True"/>
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
           <port id="0" precision="INT64">
               <dim>7</dim>
               <dim>256</dim>
               <dim>200</dim>
               <dim>10</dim>
           </port>
       </output>
   </layer>
