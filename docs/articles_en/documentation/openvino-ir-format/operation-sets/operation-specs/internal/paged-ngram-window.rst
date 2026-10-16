PagedNGramWindow
================


.. meta::
  :description: Learn about PagedNGramWindow - prepares the n-gram window of every token of several
                sequences packed into one batch, keeping the previous tokens of every sequence in a
                paged table, for NGramHashFunction with paged attention.

**Versioned name**: *PagedNGramWindow*

**Category**: *Internal*

**Short description**: *PagedNGramWindow* prepares, for every token of several sequences packed into one
batch, its n-gram window of ``ngram_size`` token ids: the token and its ``ngram_size - 1`` previous
tokens, with the document boundaries applied. It reads the previous tokens of every sequence from a
paged history table and updates the table with the new tokens. The windows are the input of
:doc:`NGramHashFunction <ngram-hash-function>` (PLE of Qwen3.8-Flash-Next) with paged attention.

**Detailed description**:

The window of a token is ``[back(ngram_size - 1), ..., back(1), back(0)]``, oldest first, where
``back(0)`` is the token itself and ``back(k)`` the token ``k`` steps back in its sequence. A token
before the start of the sequence, or before an EOS token, is replaced by ``eos_token_id``, so that the
n-grams of packed documents never cross an EOS: ``back(k)`` is EOS if an EOS lies strictly between it
and the token. The windows equal the ones the SDPA flow prepares with regular operations (``make_ngrams``
of *NGramHashFunction*), independently of how the tokens of a sequence are split into inference calls.

The history of a sequence, its last ``ngram_size - 1`` tokens, is a state like the convolution state of
:doc:`PagedCausalConv1D <paged-causal-conv1d>`, and it is managed in the same way: one row of
``history_table`` holds one state snapshot, and the rows of every sequence are given by
``la_block_indices`` / ``la_block_indices_begins``.

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

The graph flow of the PLE lookup is *PagedNGramWindow* -> *NGramHashFunction* -> *Gather* (``table``,
``row_ids``, axis 0) -> *Reshape* ``[T, H * head_dim]``.

*PagedNGramWindow* provides functionality according to the following pseudo-code using ``numpy``:

.. code-block:: py
   :force:

    def PagedNGramWindow(token_ids, history_table, subsequence_begins, la_block_indices, la_block_indices_begins,
                         num_processed_tokens, cache_interval, *, ngram_size, eos_token_id):
        n, c = ngram_size, ngram_size - 1
        ngrams = numpy.full((token_ids.shape[0], n), eos_token_id, dtype=token_ids.dtype)
        for s in range(len(num_processed_tokens)):
            t_begin, t_end = subsequence_begins[s], subsequence_begins[s + 1]
            blocks = la_block_indices[la_block_indices_begins[s]:la_block_indices_begins[s + 1]]
            done, ci = num_processed_tokens[s], cache_interval[s]
            # 1. the previous c tokens: EOS for a new sequence, else the state in the first block
            history = [eos_token_id] * c if done == 0 else [int(x) for x in history_table[blocks[0]]]
            seq = history + [int(x) for x in token_ids[t_begin:t_end]]
            for t in range(t_end - t_begin):
                # 2. the window of the token, oldest first: back(k) is EOS if an EOS lies strictly between
                window = seq[t:t + n]
                for k in range(1, c):
                    if window[n - 1 - k] == eos_token_id:
                        window[:n - 1 - k] = [eos_token_id] * (n - 1 - k)
                        break
                ngrams[t_begin + t] = window
                # 3. snapshots of the history every cache_interval tokens
                a = done + t                                    # absolute position of the token
                if ci > 0 and (a + 1) % ci == 0:
                    history_table[blocks[(a + 1) // ci - done // ci]] = seq[t + 1:t + 1 + c]
            # 4. the current history in the last block
            history_table[blocks[-1]] = seq[-c:]
        return ngrams

Properties that follow from the definition:

* For every sequence, ``ngrams`` equals the windows of ``make_ngrams`` of
  :doc:`NGramHashFunction <ngram-hash-function>` over all tokens of the sequence from its start,
  independently of how the tokens were split into inference calls. *NGramHashFunction* of these windows
  gives the row indices of the reference model.
* After the call, ``history_table[blocks[-1]]`` holds the last ``ngram_size - 1`` tokens of the
  sequence, and every snapshot row holds the last ``ngram_size - 1`` tokens at its position.
* A rollback of the last tokens (for example rejected speculative or MTP draft tokens) or a prefix
  cache hit resumes from a snapshot row: the history depends only on the last ``ngram_size - 1``
  tokens.
* The operation only moves token ids; there is no arithmetic. The hash, with its 64-bit arithmetic,
  stays in *NGramHashFunction*.


**Attributes**

* *ngram_size*

  * **Description**: the size of the window of every token, the largest n-gram order of the
    following *NGramHashFunction*. ``history_table`` has ``ngram_size - 1`` columns.
  * **Range of values**: an integer ``>= 2``
  * **Type**: ``int``
  * **Required**: *yes*

* *eos_token_id*

  * **Description**: the token that ends a document. Previous tokens before the start of the
    sequence or before an EOS are replaced by it.
  * **Range of values**: a non-negative integer
  * **Type**: ``int``
  * **Required**: *yes*


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

* **0**: ``ngrams`` - 2D tensor of type *T_IND* and shape ``[T, ngram_size]``: the n-gram window of
  every token, oldest first, the token itself last; the input of *NGramHashFunction*.

.. note::

   ``history_table`` (input 1) is updated in place as a side effect.


**Types**

* *T_IND*: ``int32``.


**Dimensions**

* ``T`` - total number of tokens of the batch.

* ``B_seq`` - number of sequences of the batch.


**Example**

*Qwen3.8-Flash-Next, two sequences with 3 and 2 tokens*

.. code-block:: xml
   :force:

    <layer id="1" name="paged_ple_window" type="PagedNGramWindow" version="extension">
        <data ngram_size="3" eos_token_id="248044"/>
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
            <port id="7" precision="I32"> <!-- ngrams -> NGramHashFunction -->
                <dim>5</dim>    <!-- T -->
                <dim>3</dim>    <!-- ngram_size -->
            </port>
        </output>
    </layer>
