## GatherTree <a name="GatherTree"></a> {#openvino_docs_ops_movement_GatherTree_1}

**Versioned name**: *GatherTree-1*

**Category**: Beam search post-processing

**Short description**: Generates the complete beams from the ids per each step and the parent beam ids.

**Detailed description**

The GatherTree operation implements the same algorithm as the [GatherTree operation in TensorFlow](https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq/gather_tree).

Pseudo code:

```python
final_idx[ :, :, :] = end_token
for batch in range(BATCH_SIZE):
    for beam in range(BEAM_WIDTH):
        max_sequence_in_beam = min(MAX_TIME, max_seq_len[batch])

        parent = parent_idx[max_sequence_in_beam - 1, batch, beam]

        final_idx[max_sequence_in_beam - 1, batch, beam] = step_idx[max_sequence_in_beam - 1, batch, beam]

        for level in reversed(range(max_sequence_in_beam - 1)):
            final_idx[level, batch, beam] = step_idx[level, batch, parent]

            parent = parent_idx[level, batch, parent]

        # For a given beam, past the time step containing the first decoded end_token
        # all values are filled in with end_token.
        finished = False
        for time in range(max_sequence_in_beam):
            if(finished):
                final_idx[time, batch, beam] = end_token
            elif(final_idx[time, batch, beam] == end_token):
                finished = True
```

Element data types for all input tensors should match each other.

**Attributes**: *GatherTree* has no attributes

**Inputs**

* **1**:  `step_ids` -- a tensor of shape `[MAX_TIME, BATCH_SIZE, BEAM_WIDTH]` of type *T* with indices from per each step. **Required.**

* **2**:  `parent_idx` -- a tensor of shape `[MAX_TIME, BATCH_SIZE, BEAM_WIDTH]` of type *T* with parent beam indices. **Required.**

* **3**:  `max_seq_len` -- a tensor of shape `[BATCH_SIZE]` of type *T* with maximum lengths for each sequence in the batch. **Required.**

* **4**:  `end_token` -- a scalar tensor of type *T* with value of the end marker in a sequence. **Required.**


**Outputs**

* **1**: `final_idx` -- a tensor of shape `[MAX_TIME, BATCH_SIZE, BEAM_WIDTH]` of type *T*.

**Types**

* *T*: `float32` or `int32`; `float32` should have integer values only.

**Example**

```xml
<layer type="GatherTree" ...>
    <input>
        <port id="0">
            <dim>100</dim>
            <dim>1</dim>
            <dim>10</dim>
        </port>
        <port id="1">
            <dim>100</dim>
            <dim>1</dim>
            <dim>10</dim>
        </port>
        <port id="2">
            <dim>1</dim>
        </port>
        <port id="3">
        </port>
    </input>
    <output>
        <port id="0">
            <dim>100</dim>
            <dim>1</dim>
            <dim>10</dim>
        </port>
    </output>
</layer>
```
