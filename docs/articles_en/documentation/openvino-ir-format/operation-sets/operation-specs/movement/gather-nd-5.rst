GatherND
========



.. meta::
  :description: Learn about GatherND-5 - a data movement operation,
                which can be performed on two required input tensors.

**Versioned name**: *GatherND-5*

**Category**: *Data movement*

**Short description**: *GatherND* gathers slices from input tensor into a tensor of a shape specified by indices.

**Detailed description**: *GatherND* gathers slices from ``data`` by ``indices`` and forms a tensor of a shape specified by ``indices``.

``indices`` is ``K``-dimensional integer tensor or ``K-1``-dimensional tensor of tuples with indices by which
the operation gathers elements or slices from ``data`` tensor. A position ``i_0, ..., i_{K-2}`` in the ``indices``
tensor corresponds to a tuple with indices ``indices[i_0, ..., i_{K-2}]`` of a length equal to ``indices.shape[-1]``.
By this tuple with indices the operation gathers a slice or an element from ``data`` tensor and insert it into the
output at position ``i_0, ..., i_{K-2}`` as the following formula:

output[i_0, ..., i_{K-2},:,...,:] = data[indices[i_0, ..., i_{K-2}],:,...,:]

The last dimension of `indices` tensor must be not greater than a rank of `data` tensor, i.e. `indices.shape[-1] <= data.rank`.
The shape of the output can be computed as `indices.shape[:-1] + data.shape[indices.shape[-1]:]`.

Example 1 shows how *GatherND* operates with elements from `data` tensor:

.. code-block:: sh

   indices = [[0, 0],
              [1, 0]]
   data    = [[1, 2],
              [3, 4]]
   output  = [1, 3]


Example 2 shows how *GatherND* operates with slices from ``data`` tensor:

.. code-block:: sh

   indices = [[1], [0]]
   data    = [[1, 2],
              [3, 4]]
   output  = [[3, 4],
              [1, 2]]


Example 3 shows how *GatherND* operates when `indices` tensor has leading dimensions:

.. code-block:: sh

   indices = [[[1]], [[0]]]
   data    = [[1, 2],
              [3, 4]]
   output  = [[[3, 4]],
              [[1, 2]]]


**Attributes**:

* *batch_dims*

  * **Description**: *batch_dims* (denoted as ``b``) is a leading number of dimensions of ``data`` tensor
    and ``indices`` representing the batches, and *GatherND* starts to gather from the ``b+1`` dimension.
    It requires the first ``b`` dimensions in ``data`` and ``indices`` tensors to be equal.
    In case of non-default value for *batch_dims*, the output shape is calculated as
    ``(multiplication of indices.shape[:b]) + indices.shape[b:-1] + data.shape[(indices.shape[-1] + b):]``.

    .. note::

       The calculation of output shape is incorrect for non-default *batch_dims* value greater than one.
       For correct calculations use the :doc:`GatherND_8 <gather-8>` operation.

  * **Range of values**: integer number and belongs to ``[0; min(data.rank, indices.rank))``
  * **Type**: int
  * **Default value**: 0
  * **Required**: *no*

Example 4 shows how *GatherND* operates gathering elements for non-default *batch_dims* value:

.. code-block:: sh

   batch_dims = 1
   indices = [[1],    <--- this is applied to the first batch
              [0]]    <--- this is applied to the second batch, shape = (2, 1)
   data    = [[1, 2], <--- the first batch
              [3, 4]] <--- the second batch, shape = (2, 2)
   output  = [2, 3], shape = (2)


Example 5 shows how *GatherND* operates gathering slices for non-default *batch_dims* value:

.. code-block:: sh

   batch_dims = 1
   indices = [[1], <--- this is applied to the first batch
              [0]] <--- this is applied to the second batch, shape = (2, 1)
   data    = [[[1,   2,  3,  4], [ 5,  6,  7,  8], [ 9, 10, 11, 12]]  <--- the first batch
              [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]] <--- the second batch, shape = (2, 3, 4)
   output  = [[ 5,  6,  7,  8], [13, 14, 15, 16]], shape = (2, 4)


More complex, example 6 shows how *GatherND* operates gathering slices with leading dimensions
for non-default *batch_dims* value:

.. code-block:: sh

   batch_dims = 2
   indices = [[[[1]], <--- this is applied to the first batch
               [[0]],
               [[2]]],
              [[[0]],
               [[2]],
               [[2]]] <--- this is applied to the sixth batch
             ], shape = (2, 3, 1, 1)
   data    = [[[1,   2,  3,  4], <--- this is the first batch
               [ 5,  6,  7,  8],
               [ 9, 10, 11, 12]]
              [[13, 14, 15, 16],
               [17, 18, 19, 20],
               [21, 22, 23, 24]] <--- this is the sixth batch
             ] <--- the second batch, shape = (2, 3, 4)
   output  = [[2], [5], [11], [13], [19], [23]], shape = (6, 1)


**Inputs**:

* **1**: ``data`` tensor of type *T*. This is a tensor of a rank not lower than 1. **Required.**
* **2**: ``indices`` tensor of type *T_IND*. This is a tensor of a rank not lower than 1.
  It requires that all indices from this tensor are be in the range of ``[0, s-1]`` where ``s`` is
  corresponding dimension to which this index is applied. **Required**.

**Outputs**:

* **1**: Tensor with gathered values of type *T*.

**Types**

* *T*: any supported type.
* *T_IND*: any supported integer types.

**Examples**

.. code-block:: xml
   :force:

   <layer id="1" type="GatherND">
       <data batch_dims=0 />
       <input>
           <port id="0">
               <dim>1000</dim>
               <dim>256</dim>
               <dim>10</dim>
               <dim>15</dim>
           </port>
           <port id="1">
               <dim>25</dim>
               <dim>125</dim>
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="3">
               <dim>25</dim>
               <dim>125</dim>
               <dim>15</dim>
           </port>
       </output>
   </layer>


.. code-block:: xml
   :force:

   <layer id="1" type="GatherND">
       <data batch_dims=2 />
       <input>
           <port id="0">
               <dim>30</dim>
               <dim>2</dim>
               <dim>100</dim>
               <dim>35</dim>
           </port>
           <port id="1">
               <dim>30</dim>
               <dim>2</dim>
               <dim>3</dim>
               <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="3">
               <dim>60</dim>
               <dim>3</dim>
               <dim>35</dim>
           </port>
       </output>
   </layer>



