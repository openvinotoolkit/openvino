EmbeddingSegmentsSum
====================


.. meta::
  :description: Learn about EmbeddingSegmentsSum-3 - a sparse operation, which
                can be performed on four required and two optional input tensors.

**Versioned name**: *EmbeddingSegmentsSum-3*

**Category**: *Sparse*

**Short description**: Computes sums of segments of embeddings, without instantiating the intermediate embeddings.

**Detailed description**: This is `sparse.segment_sum <https://www.tensorflow.org/api_docs/python/tf/sparse/segment_sum>`__ operation from Tensorflow. For each index in ``indices`` this operator gets values from ``data`` embedding table and sums all values belonging to each segment. Values in ``segment_ids`` define which segment index in ``indices`` tensor belong to, e.g. ``segments_ids`` with value ``[0,0,0,1,1,3,5,5]`` define 4 non empty segments other segments are empty, the number of segments is defined by ``num_segments`` input.

**Attributes**: EmbeddingSegmentsSum operation has no attributes.

**Inputs**:

* **1**: ``emb_table`` tensor containing the embedding lookup table of the module of shape ``[num_emb, emb_dim1, emb_dim2, ...]`` and of type *T*. **Required.**
* **2**: ``indices`` tensor of shape ``[num_indices]`` and of type *T_IND*. **Required.**
* **3**: ``segment_ids`` tensor of shape ``[num_indices]`` and of type *T_IND* with indices into the output Tensor. Values should be sorted and can be repeated. **Required.**
* **4**: ``num_segments`` scalar of type *T_IND* indicating the number of segments. **Required.**
* **5**: ``default_index`` scalar of type *T_IND* containing default index in embedding table to fill empty segments. If not provided empty segments are filled with zeros. **Optional.**
* **6**: ``per_sample_weights`` tensor of the same shape as ``indices`` and of type *T*. Each value in this tensor are multiplied with each value pooled from embedding table for each index. Optional, default is tensor of ones.

**Outputs**:

* **1**: tensor of shape ``[num_segments, emb_dim1, emb_dim2, ...]`` and of type *T* containing embeddings for each bag.

**Types**

* *T*: any numeric type.
* *T_IND*: ``int32`` or ``int64``.

**Example**

.. code-block:: cpp

   <layer ... type="EmbeddingSegmentsSum" ... >
       <input>
           <port id="0">     <!-- emb_table value is: [[-0.2, -0.6], [-0.1, -0.4], [-1.9, -1.8], [-1.,  1.5], [ 0.8, -0.7]] -->
               <dim>5</dim>
               <dim>2</dim>
           </port>
           <port id="1">     <!-- indices value is: [0, 2, 3, 4] -->
               <dim>4</dim>
           </port>
           <port id="2"/>    <!-- segment_ids value is: [0, 0, 2, 2] - second segment is empty -->
               <dim>4</dim>
           </port>
           <port id="3"/>    <!-- num_segments value is: 3 -->
           <port id="4"/>    <!-- default_index value is: 0 -->
           <port id="5"/>    <!-- per_sample_weigths value is: [0.5, 0.5, 0.5, 0.5] -->
               <dim>4</dim>
           </port>
       </input>
       <output>
           <port id="6">     <!-- output value is: [[-1.05, -1.2], [-0.2, -0.6], [-0.1, 0.4]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </output>
   </layer>


