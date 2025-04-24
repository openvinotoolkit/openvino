EmbeddingBagPackedSum
=====================


.. meta::
  :description: Learn about EmbeddingBagPackedSum-3 - a sparse operation, which
                can be performed on two required and one optional input tensor.

**Versioned name**: *EmbeddingBagPackedSum-3*

**Category**: *Sparse*

**Short description**: Computes sums of "bags" of embeddings, without instantiating the intermediate embeddings.

**Detailed description**:

Operation EmbeddingBagPackedSum is an implementation of ``torch.nn.EmbeddingBag`` in ``sum`` mode, which indices input being 2D tensor of shape ``[batch, indices_per_bag]``.
Operation is equivalent to *ReduceSum(Multiply(Gather(emb_table, indices, axis=0), Unsqueeze(per_sample_weights, -1)), axis=1)*.

**Attributes**: EmbeddingBagPackedSum operation has no attributes.

**Inputs**:

* **1**: ``emb_table`` tensor containing the embedding lookup table of the module of shape ``[num_emb, emb_dim1, emb_dim2, ...]`` and of type *T*. **Required.**
* **2**: ``indices`` tensor of shape ``[batch, indices_per_bag]`` and of type *T_IND*. **Required.**
* **3**: ``per_sample_weights`` tensor of the same shape as ``indices`` and of type *T*. Each value in this tensor are multiplied with each value pooled from embedding table for each index. Optional, default is tensor of ones.

**Outputs**:

* **1**: tensor of shape ``[batch, emb_dim1, emb_dim2, ...]`` and of type *T* containing embeddings for each bag.

**Types**

* *T*: any numeric type.
* *T_IND*: ``int32`` or ``int64``.

**Example**

.. code-block:: xml

   <layer ... type="EmbeddingBagPackedSum" ... >
       <input>
           <port id="0">     <!-- emb_table value is: [[-0.2, -0.6], [-0.1, -0.4], [-1.9, -1.8], [-1.,  1.5], [ 0.8, -0.7]] -->
               <dim>5</dim>
               <dim>2</dim>
           </port>
           <port id="1">     <!-- indices value is: [[0, 2], [1, 2], [3, 4]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
           <port id="2"/>    <!-- per_sample_weights value is: [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="3">     <!-- output value is: [[-1.05, -1.2], [-1., -1.1], [-0.1, 0.4]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </output>
   </layer>


