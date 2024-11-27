EmbeddingBagPacked
=====================


.. meta::
  :description: Learn about EmbeddingBagPacked-15 - a sparse operation, which
                can be performed on two required and one optional input tensor.

**Versioned name**: *EmbeddingBagPacked-15*

**Category**: *Sparse*

**Short description**: Computes sums or means of "bags" of embeddings, without instantiating the intermediate embeddings.

**Detailed description**:

Operation EmbeddingBagPacked is an implementation of ``torch.nn.EmbeddingBag`` with indices input being 2D tensor of shape ``[batch, indices_per_bag]``.
Operation is equivalent to *gather_op = Gather(emb_table, indices, axis=0)* followed by reduction:

  * *sum* - *ReduceSum(Multiply(gather_op, Unsqueeze(per_sample_weights, -1)), axis=1)*,
  * *mean* - *ReduceMean(gather_op, axis=1)*.

**Attributes**:

* *reduction*

  * **Description**: reduction mode.
  * **Range of values**:

    * sum - compute weighted sum, using corresponding values of ``per_sample_weights`` as weights if provided.
    * mean - compute average of values in bag. Input ``per_sample_weights`` is not supported and will raise exception.

  * **Type**: ``string``
  * **Default value**: sum
  * **Required**: *no*

**Inputs**:

* **1**: ``emb_table`` tensor containing the embedding lookup table of the module of shape ``[num_emb, emb_dim1, emb_dim2, ...]`` and of type *T*. **Required.**
* **2**: ``indices`` tensor of shape ``[batch, indices_per_bag]`` and of type *T_IND*. **Required.**
* **3**: ``per_sample_weights`` tensor of the same shape as ``indices`` and of type *T* supported only in ``sum`` mode. Each value in this tensor are multiplied with each value pooled from embedding table for each index. Optional, default is tensor of ones. **Optional.**

**Outputs**:

* **1**: tensor of shape ``[batch, emb_dim1, emb_dim2, ...]`` and of type *T* containing embeddings for each bag.

**Types**

* *T*: any numeric type.
* *T_IND*: ``int32`` or ``int64``.

**Example**

*Example 1: reduction set to sum, per_sample_weights are not provided.*

.. code-block:: xml

   <layer ... type="EmbeddingBagPacked" ... >
       <data reduction="sum"/>
       <input>
           <port id="0">     <!-- emb_table value is: [[-0.2, -0.6], [-0.1, -0.4], [-1.9, -1.8], [-1.,  1.5], [ 0.8, -0.7]] -->
               <dim>5</dim>
               <dim>2</dim>
           </port>
           <port id="1">     <!-- indices value is: [[0, 2], [1, 2], [3, 4]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="2">     <!-- output value is: [[-2.1, -2.4], [-2., -2.2], [-0.2, 0.8]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </output>
   </layer>

*Example 2: reduction set to sum and per_sample_weights are provided.*

.. code-block:: xml

   <layer ... type="EmbeddingBagPacked" ... >
       <data reduction="sum"/>
       <input>
           <port id="0">     <!-- emb_table value is: [[-0.2, -0.6], [-0.1, -0.4], [-1.9, -1.8], [-1.,  1.5], [ 0.8, -0.7]] -->
               <dim>5</dim>
               <dim>2</dim>
           </port>
           <port id="1">     <!-- indices value is: [[0, 2], [1, 2], [3, 4]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
           <port id="2"/>    <!-- per_sample_weights value is: [[0.5, 0.5], [0.3, 0.7], [2., -1.]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="3">     <!-- output value is: [[-1.05, -1.2], [-1.36, -1.38], [-2.8, 3.7]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </output>
   </layer>

*Example 3: reduction set to mean, per_sample_weights are not provided.*

.. code-block:: xml

   <layer ... type="EmbeddingBagPacked" ... >
       <data reduction="mean"/>
       <input>
           <port id="0">     <!-- emb_table value is: [[-0.2, -0.6], [-0.1, -0.4], [-1.9, -1.8], [-1.,  1.5], [ 0.8, -0.7]] -->
               <dim>5</dim>
               <dim>2</dim>
           </port>
           <port id="1">     <!-- indices value is: [[0, 2], [1, 2], [3, 4]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="2">     <!-- output value is: [[-1.05, -1.2], [-1., -1.1], [-0.1, 0.4]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </output>
   </layer>

