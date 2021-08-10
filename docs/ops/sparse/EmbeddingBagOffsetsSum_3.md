## EmbeddingBagOffsetsSum <a name="EmbeddingBagOffsetsSum"></a> {#openvino_docs_ops_sparse_EmbeddingBagOffsetsSum_3}

**Versioned name**: *EmbeddingBagOffsetsSum-3*

**Category**: *Sparse*

**Short description**: Computes sums of "bags" of embeddings, without instantiating the intermediate embeddings.

**Detailed description**: This is the second case of the PyTorch [EmbeddingBag](https://pytorch.org/docs/stable/nn.html#embeddingbag), it has indices in two 1D tensors provided as 2nd and 3rd inputs. For each index in `indices` this operator gets values from `data` embedding table and sums all values belonging to each bag. Values in `offsets` define starting index in `indices` tensor of each "bag", e.g. `offsets` with value `[0,3,4,4,6]` define 5 "bags" containing `[3,1,0,2,n-6]` elements.

**Inputs**:

*   **1**: `emb_table` tensor containing the embedding lookup table of the module of shape `[num_emb, emb_dim1, emb_dim2, ...]` and  of type *T*. **Required.**

*   **2**: `indices` tensor of shape `[num_indices]` and of type *T_IND*. **Required.**

*   **3**: `offsets` tensor of shape `[batch]` and of type *T_IND* containing the starting index positions of each "bag" in `indices`. **Required.**

*   **4**: `default_index` scalar of type *T_IND* containing default index in embedding table to fill empty "bags". If not provided empty "bags" are filled with zeros. **Optional.**

*   **5**: `per_sample_weights` tensor of the same shape as `indices` and of type *T*. Each value in this tensor are multiplied with each value pooled from embedding table for each index. Optional, default is tensor of ones.

**Outputs**:

*   **1**: tensor of shape `[batch, emb_dim1, emb_dim2, ...]` and of type *T* containing embeddings for each bag.

**Types**

* *T*: any numeric type.

* *T_IND*: `int32` or `int64`.

**Example**

```xml
<layer ... type="EmbeddingBagOffsetsSum" ... >
    <input>
        <port id="0">     <!-- emb_table value is: [[-0.2, -0.6], [-0.1, -0.4], [-1.9, -1.8], [-1.,  1.5], [ 0.8, -0.7]] -->
            <dim>5</dim>
            <dim>2</dim>
        </port>
        <port id="1">     <!-- indices value is: [0, 2, 3, 4] -->
            <dim>4</dim>
        </port>
        <port id="2">     <!-- offsets value is: [0, 2, 2] - second "bag" is empty -->
            <dim>3</dim>
        </port>
        <port id="3"/>    <!-- default_index value is: 0 -->
        <port id="4"/>    <!-- per_sample_weigths value is: [0.5, 0.5, 0.5, 0.5] -->
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
```
