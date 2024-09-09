EmbeddingBagOffsets
======================


.. meta::
  :description: Learn about EmbeddingBagOffsets-15 - a sparse operation, which
                can be performed on three required and two optional input tensors.

**Versioned name**: *EmbeddingBagOffsets-15*

**Category**: *Sparse*

**Short description**: Computes sums or means of "bags" of embeddings, without instantiating the intermediate embeddings.

**Detailed description**:

Operation EmbeddingBagOffsets is an implementation of ``torch.nn.EmbeddingBag`` with indices and offsets inputs being 1D tensors.

For each index in ``indices`` this operator gathers values from ``emb_table`` embedding table. Then values at indices in the range of the same bag (based on ``offset`` input) are reduced according to ``reduction`` attribute.

Values in ``offsets`` define starting index in ``indices`` tensor of each "bag",
e.g. ``offsets`` with value ``[0, 3, 4, 4, 6]`` define 5 "bags" containing ``[3, 1, 0, 2, num_indices-6]`` elements corresponding to ``[indices[0:3], indices[3:4], empty_bag, indices[4:6], indices[6:]]`` slices of indices per bag.

EmbeddingBagOffsets is an equivalent to following NumPy snippet:

.. code-block:: py

    def embedding_bag_offsets(
        emb_table: np.ndarray,
        indices: np.ndarray,
        offsets: np.ndarray,
        default_index: Optional[int] = None,
        per_sample_weights: Optional[np.ndarray] = None,
        reduction: Literal["sum", "mean"] = "sum",
    ):
        assert (
            reduction == "sum" or per_sample_weights is None
        ), "Attribute per_sample_weights is only supported in sum reduction."
        if per_sample_weights is None:
            per_sample_weights = np.ones_like(indices)
        embeddings = []
        for emb_idx, emb_weight in zip(indices, per_sample_weights):
            embeddings.append(emb_table[emb_idx] * emb_weight)
        previous_offset = offsets[0]
        bags = []
        offsets = np.append(offsets, len(indices))
        for bag_offset in offsets[1:]:
            bag_size = bag_offset - previous_offset
            if bag_size != 0:
                embedding_bag = embeddings[previous_offset:bag_offset]
                reduced_bag = np.add.reduce(embedding_bag)
                if reduction == "mean":
                    reduced_bag = reduced_bag / bag_size
                bags.append(reduced_bag)
            else:
                # Empty bag case
                if default_index is not None and default_index != -1:
                    bags.append(emb_table[default_index])
                else:
                    bags.append(np.zeros(emb_table.shape[1:]))
            previous_offset = bag_offset
        return np.stack(bags, axis=0)


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

* **1**: ``emb_table`` tensor containing the embedding lookup table of the module of shape ``[num_emb, emb_dim1, emb_dim2, ...]`` and  of type *T*. **Required.**
* **2**: ``indices`` tensor of shape ``[num_indices]`` and of type *T_IND*. **Required.**
* **3**: ``offsets`` tensor of shape ``[batch]`` and of type *T_IND* containing the starting index positions of each "bag" in ``indices``. Maximum value of offsets cannot be greater than length of ``indices``. **Required.**
* **4**: ``default_index`` scalar of type *T_IND* containing default index in embedding table to fill empty "bags". If set to ``-1`` or not provided, empty "bags" are filled with zeros. Reverse indexing using negative values is not supported. **Optional.**
* **5**: ``per_sample_weights`` tensor of the same shape as ``indices`` and of type *T*. Supported only when *reduction* attribute is set to ``"sum"``. Each value in this tensor are multiplied with each value pooled from embedding table for each index. Optional, default is tensor of ones. **Optional.**

**Outputs**:

* **1**: tensor of shape ``[batch, emb_dim1, emb_dim2, ...]`` and of type *T* containing embeddings for each bag.

**Types**

* *T*: any numeric type.
* *T_IND*: ``int32`` or ``int64``.

**Example**

*Example 1: per_sample_weights are provided, default_index is set to 0 to fill empty bag with values gathered form emb_table on given index.*

.. code-block:: xml

   <layer ... type="EmbeddingBagOffsets" ... >
       <data reduction="sum"/>
       <input>
           <port id="0">     <!-- emb_table value is: [[-0.2, -0.6], [-0.1, -0.4], [-1.9, -1.8], [-1.,  1.5], [ 0.8, -0.7]] -->
               <dim>5</dim>
               <dim>2</dim>
           </port>
           <port id="1">     <!-- indices value is: [0, 2, 3, 4] -->
               <dim>4</dim>
           </port>
           <port id="2">     <!-- offsets value is: [0, 2, 2] - 3 "bags" containing [2,0,4-2] elements, second "bag" is empty -->
               <dim>3</dim>
           </port>
           <port id="3"/>    <!-- default_index value is: 0 -->
           <port id="4"/>    <!-- per_sample_weights value is: [0.5, 0.5, 0.5, 0.5] -->
               <dim>4</dim>
           </port>
       </input>
       <output>
           <port id="5">     <!-- output value is: [[-1.05, -1.2], [-0.2, -0.6], [-0.1, 0.4]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </output>
   </layer>

*Example 2: per_sample_weights are provided, default_index is set to -1 to fill empty bag with 0.*

.. code-block:: xml

   <layer ... type="EmbeddingBagOffsets" ... >
       <data reduction="sum"/>
       <input>
           <port id="0">     <!-- emb_table value is: [[-0.2, -0.6], [-0.1, -0.4], [-1.9, -1.8], [-1.,  1.5], [ 0.8, -0.7]] -->
               <dim>5</dim>
               <dim>2</dim>
           </port>
           <port id="1">     <!-- indices value is: [0, 2, 3, 4] -->
               <dim>4</dim>
           </port>
           <port id="2">     <!-- offsets value is: [0, 2, 2] - 3 "bags" containing [2,0,4-2] elements, second "bag" is empty -->
               <dim>3</dim>
           </port>
           <port id="3"/>    <!-- default_index value is: -1 - fill empty bag with 0-->
           <port id="4"/>    <!-- per_sample_weights value is: [0.5, 0.2, -2, 1] -->
               <dim>4</dim>
           </port>
       </input>
       <output>
           <port id="5">     <!-- output value is: [[-0.48, -0.66], [0., 0.], [2.8, -3.7]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </output>
   </layer>

*Example 3: Example of reduction set to mean.*

.. code-block:: xml

   <layer ... type="EmbeddingBagOffsets" ... >
       <data reduction="mean"/>
       <input>
           <port id="0">     <!-- emb_table value is: [[-0.2, -0.6], [-0.1, -0.4], [-1.9, -1.8], [-1.,  1.5], [ 0.8, -0.7]] -->
               <dim>5</dim>
               <dim>2</dim>
           </port>
           <port id="1">     <!-- indices value is: [0, 2, 3, 4] -->
               <dim>4</dim>
           </port>
           <port id="2">     <!-- offsets value is: [0, 2, 2] - 3 "bags" containing [2,0,4-2] elements, second "bag" is empty -->
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="3">     <!-- output value is: [[-1.05, -1.2], [0., 0.], [-0.1, 0.4]] -->
               <dim>3</dim>
               <dim>2</dim>
           </port>
       </output>
   </layer>
