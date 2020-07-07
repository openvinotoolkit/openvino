## GatherTree <a name="GatherTree"></a>

**Versioned name**: *GatherTree-1*

**Category**: Beam search post-processing

**Short description**: Generates the complete beams from the ids per each step and the parent beam ids.

**Detailed description**

GatherTree operation implements the same algorithm as GatherTree operation in TensorFlow. Please see complete documentation [here](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/contrib/seq2seq/gather_tree?hl=en).

Pseudo code:

```python
for batch in range(BATCH_SIZE):
    for beam in range(BEAM_WIDTH):
        max_sequence_in_beam = min(MAX_TIME, max_seq_len[batch])

        parent = parent_idx[max_sequence_in_beam - 1, batch, beam]

        for level in reversed(range(max_sequence_in_beam - 1)):
            final_idx[level, batch, beam] = step_idx[level, batch, parent]

            parent = parent_idx[level, batch, parent]
```

Element data types for all input tensors should match each other.

**Attributes**: *GatherTree* has no attributes

**Inputs**

* **1**:  `step_ids` -- a tensor of shape `[MAX_TIME, BATCH_SIZE, BEAM_WIDTH]` of type `T` with indices from per each step. Required.

* **2**:  `parent_idx` -- a tensor of shape `[MAX_TIME, BATCH_SIZE, BEAM_WIDTH]` of type `T` with parent beam indices. Required.

* **3**:  `max_seq_len` -- a tensor of shape `[BATCH_SIZE]` of type `T` with maximum lengths for each sequence in the batch. Required.

* **4**:  `end_token` -- a scalar tensor of type `T` with value of the end marker in a sequence. Required.


**Outputs**

* **1**: `final_idx` -- a tensor of shape `[MAX_TIME, BATCH_SIZE, BEAM_WIDTH]` of type `T`.

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
