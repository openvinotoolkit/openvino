PagedQSAHashFunction
====================


.. meta::
  :description: Learn about PagedQSAHashFunction - the paged variant of QSAHashFunction that keeps
                the previous tokens of every sequence in a paged table and updates it with the new
                tokens.

**Versioned name**: *PagedQSAHashFunction*

**Category**: *Internal*

**Short description**: *PagedQSAHashFunction* is the paged variant of
:doc:`QSAHashFunction <qsa-hash-function>`. It hashes the n-grams of the tokens of several sequences
packed into one batch into row indices of the packed n-gram embedding table (PLE of
Qwen3.8-Flash-Next), reads the previous tokens of every sequence from a paged history table and
updates the table with the new tokens.

**Detailed description**:

The hash is the one of *QSAHashFunction*. The history of a sequence, its last ``ngram_size - 1``
tokens, is a state like the convolution state of :doc:`PagedCausalConv1D <paged-causal-conv1d>`, and
it is managed in the same way: one row of ``history_table`` holds one state snapshot, and the rows of
every sequence are given by ``la_block_indices`` / ``la_block_indices_begins``.

**Paged state management**

For the sequence ``s`` with the logical blocks
``blocks = la_block_indices[la_block_indices_begins[s] : la_block_indices_begins[s + 1]]``:

* ``blocks[0]`` is read: it holds the history after the ``num_processed_tokens[s]`` tokens already
  processed. A new sequence (``num_processed_tokens[s] == 0``) starts with ``ngram_size - 1`` EOS
  tokens and the table row is not read.
* With ``cache_interval[s] > 0``, the history after every ``cache_interval[s]``-th token (counted
  from the start of the sequence) is written to ``blocks[k]``, ``k = (a + 1) // cache_interval[s] -
  num_processed_tokens[s] // cache_interval[s]`` for the token at the absolute position ``a``. These
  snapshots allow prefix caching and chunked prefill to resume at a cached position.
* The history after the last new token is written to ``blocks[-1]``.

The read and the write blocks may refer to the same row, which is then updated in place; the cases
are the ones of *PagedCausalConv1D* and *PagedGatedDeltaNet*. The host (OpenVINO GenAI) allocates the
rows, together with the states of the other linear-attention layers. The table is updated in place;
its initial content does not matter, because a new sequence does not read it.

The graph flow of the PLE lookup is *PagedQSAHashFunction* -> *Gather* (``table``, ``row_ids``,
axis 0) -> *Reshape* ``[T, H * head_dim]``.

*PagedQSAHashFunction* provides functionality according to the following pseudo-code using ``numpy``
and the pseudo-code of :doc:`QSAHashFunction <qsa-hash-function>`:

.. code-block:: py
   :force:

    def PagedQSAHashFunction(token_ids, history_table, subsequence_begins, la_block_indices, la_block_indices_begins,
                             num_processed_tokens, cache_interval, *, ngram_size, heads_per_ngram, multipliers,
                             head_vocab_sizes, head_offsets, eos_token_id):
        c = ngram_size - 1
        H = c * heads_per_ngram
        T = token_ids.shape[0]
        row_ids = numpy.zeros((T, H), dtype=numpy.int64)
        for s in range(len(num_processed_tokens)):
            t_begin, t_end = subsequence_begins[s], subsequence_begins[s + 1]
            blocks = la_block_indices[la_block_indices_begins[s]:la_block_indices_begins[s + 1]]
            done, ci = num_processed_tokens[s], cache_interval[s]
            # 1. the previous c tokens: EOS for a new sequence, else the state in the first block
            history = [eos_token_id] * c if done == 0 else [int(x) for x in history_table[blocks[0]]]
            # 2. hash exactly as QSAHashFunction on one sequence
            ids, _ = QSAHashFunction(token_ids[None, t_begin:t_end], numpy.array([history]), ngram_size=ngram_size,
                                     heads_per_ngram=heads_per_ngram, multipliers=multipliers,
                                     head_vocab_sizes=head_vocab_sizes, head_offsets=head_offsets,
                                     eos_token_id=eos_token_id)
            row_ids[t_begin:t_end] = ids[0]
            # 3. update the table: snapshots every cache_interval tokens, then the current state in the last block
            seq = history + [int(x) for x in token_ids[t_begin:t_end]]
            for t in range(t_end - t_begin):
                a = done + t                                    # absolute position of the token
                if ci > 0 and (a + 1) % ci == 0:
                    history_table[blocks[(a + 1) // ci - done // ci]] = seq[t + 1:t + 1 + c]
            history_table[blocks[-1]] = seq[-c:]
        return row_ids

Properties that follow from the definition:

* For every sequence, ``row_ids`` equals the ``row_ids`` of *QSAHashFunction* over all tokens of the
  sequence from its start, independently of how the tokens were split into inference calls.
* After the call, ``history_table[blocks[-1]]`` holds the last ``ngram_size - 1`` tokens of the
  sequence, and every snapshot row holds the last ``ngram_size - 1`` tokens at its position.
* A rollback of the last tokens (for example rejected speculative or MTP draft tokens) or a prefix
  cache hit resumes from a snapshot row: the history depends only on the last ``ngram_size - 1``
  tokens.
* The 64-bit integer arithmetic stays inside the operation; ``token_ids``, ``history_table`` and
  ``row_ids`` are ``int32``.


**Attributes**

*PagedQSAHashFunction* has the attributes *ngram_size*, *heads_per_ngram*, *multipliers*,
*head_vocab_sizes*, *head_offsets* and *eos_token_id* of
:doc:`QSAHashFunction <qsa-hash-function>`, with the same meaning.


**Inputs**

.. note::

   Input ports are positional and cannot be skipped, as for *PagedAttentionExtension*.

* **0**: ``token_ids`` - 1D tensor of type *T_IND* and shape ``[T]``: the token ids of all
  sequences of the batch, packed. Ids of padded positions are ``eos_token_id``. **Required.**

* **1**: ``history_table`` - 2D tensor of type *T_IND* and shape
  ``[num_physical_blocks, ngram_size - 1]``: the paged history snapshots, oldest token first, updated
  in place. **Required.**

* **2**: ``subsequence_begins`` - 1D tensor of type *T_IND* and shape ``[B_seq + 1]``: the tokens of
  the sequence ``s`` are ``token_ids[subsequence_begins[s] : subsequence_begins[s + 1]]``.
  **Required.**

* **3**: ``la_block_indices`` - 1D tensor of type *T_IND* and shape ``[num_logical_blocks]``: the rows
  of ``history_table`` of all sequences, as for *PagedCausalConv1D*. **Required.**

* **4**: ``la_block_indices_begins`` - 1D tensor of type *T_IND* and shape ``[B_seq + 1]``: the rows
  of the sequence ``s`` are ``la_block_indices[la_block_indices_begins[s] : la_block_indices_begins[s + 1]]``.
  **Required.**

* **5**: ``num_processed_tokens`` - 1D tensor of type *T_IND* and shape ``[B_seq]``: the number of
  tokens of every sequence processed in earlier calls. ``0`` starts a new sequence. **Required.**

* **6**: ``cache_interval`` - 1D tensor of type *T_IND* and shape ``[B_seq]``: the interval in tokens
  of the history snapshots of every sequence; ``<= 0`` keeps only the current state. **Required.**


**Outputs**

* **0**: ``row_ids`` - 2D tensor of type *T_IND* and shape ``[T, H]``: the row of the packed n-gram
  table for every token and hashed head.

.. note::

   ``history_table`` (input 1) is updated in place as a side effect.


**Types**

* *T_IND*: ``int32``.


**Dimensions**

* ``T`` - total number of tokens of the batch.

* ``B_seq`` - number of sequences of the batch.

* ``H`` - number of hashed heads, ``(ngram_size - 1) * heads_per_ngram``.


**Example**

*Qwen3.8-Flash-Next, two sequences with 3 and 2 tokens*

.. code-block:: xml
   :force:

    <layer id="1" name="paged_ple_hash" type="PagedQSAHashFunction" version="extension">
        <data ngram_size="3" heads_per_ngram="8"
              multipliers="23703573157769,20109073645365,8052911324071"
              head_vocab_sizes="20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171"
              head_offsets="0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275"
              eos_token_id="248044"/>
        <input>
            <port id="0" precision="I32"> <dim>5</dim> </port>                <!-- token_ids -->
            <port id="1" precision="I32"> <dim>-1</dim> <dim>2</dim> </port>  <!-- history_table -->
            <port id="2" precision="I32"> <dim>3</dim> </port>                <!-- subsequence_begins -->
            <port id="3" precision="I32"> <dim>-1</dim> </port>               <!-- la_block_indices -->
            <port id="4" precision="I32"> <dim>3</dim> </port>                <!-- la_block_indices_begins -->
            <port id="5" precision="I32"> <dim>2</dim> </port>                <!-- num_processed_tokens -->
            <port id="6" precision="I32"> <dim>2</dim> </port>                <!-- cache_interval -->
        </input>
        <output>
            <port id="7" precision="I32"> <!-- row_ids -->
                <dim>5</dim>    <!-- T -->
                <dim>16</dim>   <!-- H -->
            </port>
        </output>
    </layer>
