QSAHashFunction
===============


.. meta::
  :description: Learn about QSAHashFunction - computes the row indices of the hashed n-gram
                embedding table (PLE) of Qwen3.8-Flash-Next with exact 64-bit integer hashing inside
                the operation.

**Versioned name**: *QSAHashFunction*

**Category**: *Sequence processing*

**Short description**: *QSAHashFunction* hashes the bigrams, trigrams and longer n-grams that end at
every token into the row indices of a packed n-gram embedding table. The 64-bit integer arithmetic of
the hash stays inside the operation; the inputs and outputs fit 32-bit integers.

**Detailed description**:

Qwen3.8-Flash-Next (``qwen4_exp`` in Hugging Face Transformers) adds a Per-Layer Embedding (PLE) at
decoder layer 2: an n-gram embedding looked up by hashes of the last tokens
(``Qwen4ExpTextNGramEmbedding``). The table has ``H = (ngram_size - 1) * heads_per_ngram`` hashed
heads, 16 for the model: heads 0-7 hash the bigram and heads 8-15 the trigram that ends at a token.
Every head has its own prime vocabulary size and its rows are packed into one table at an offset.
*QSAHashFunction* computes, for every token and head, the row of the table:

* the token and the ``ngram_size - 1`` previous tokens are taken from the token history; a previous
  token before the start of the sequence, or before an EOS token, is replaced by EOS, so that the
  n-grams of packed documents never cross an EOS;
* the n-gram of order ``o`` is mixed as ``(x[t] * m[0]) XOR (x[t-1] * m[1]) XOR ... XOR (x[t-o+1] * m[o-1])``
  with the odd multipliers ``m``;
* head ``h`` of order ``o`` takes the row ``(mix_o mod head_vocab_sizes[h]) + head_offsets[h]``.

The products of a token id and a multiplier need up to 63 bits (``token < 2^18``, multipliers
``< (2^63 - 1) / vocab_size``), and the modulo is taken of the 63-bit mix. The result rows, below the
table size of about ``3.2e8`` for the model, fit ``int32``. A device without native 64-bit integer
arithmetic in the graph (for example a plugin that converts ``i64`` to ``i32``) cannot evaluate the
hash with regular operations; *QSAHashFunction* keeps the 64-bit arithmetic inside its kernel, and
the constants as ``int64`` attributes.

The graph flow of the PLE lookup is *QSAHashFunction* -> *Gather* (``table``, ``row_ids``, axis 0)
-> *Reshape* ``[N, L, H * head_dim]``. The paged counterpart is
:doc:`PagedQSAHashFunction <paged-qsa-hash-function>`.

*QSAHashFunction* provides functionality according to the following pseudo-code using ``numpy``:

.. code-block:: py
   :force:

    def QSAHashFunction(token_ids, history, *, ngram_size, heads_per_ngram, multipliers, head_vocab_sizes,
                        head_offsets, eos_token_id):
        N, L = token_ids.shape
        c = ngram_size - 1                                      # previous tokens of an n-gram
        H = c * heads_per_ngram                                 # hashed heads
        # computed exactly in 64 bits; row_ids are returned as int32, new_history as the type of history
        row_ids = numpy.zeros((N, L, H), dtype=numpy.int64)
        new_history = numpy.zeros((N, c), dtype=numpy.int64)
        for n in range(N):
            seq = [int(x) for x in history[n]] + [int(x) for x in token_ids[n]]     # c + L tokens
            prev_eos, last = [], -1                             # the last EOS strictly before every position
            for i, x in enumerate(seq):
                prev_eos.append(last)
                if x == eos_token_id:
                    last = i
            for l in range(L):
                i = c + l

                def back(k):         # the token k steps back; EOS before the start or across an EOS
                    j = i - k
                    return seq[j] if j >= 0 and j > prev_eos[i] else eos_token_id

                mix = back(0) * int(multipliers[0])             # exact integer math, < 2**63
                for o in range(2, ngram_size + 1):              # bigram, trigram, ...
                    mix ^= back(o - 1) * int(multipliers[o - 1])
                    for e in range(heads_per_ngram):
                        h = (o - 2) * heads_per_ngram + e
                        row_ids[n, l, h] = mix % int(head_vocab_sizes[h]) + int(head_offsets[h])
            new_history[n] = seq[-c:]                           # the last c tokens, for the next call
        return row_ids.astype(numpy.int32), new_history.astype(history.dtype)

Properties that follow from the definition:

* The result equals the row indices of ``Qwen4ExpTextNGramEmbedding.forward``: the history plays the
  role of the cached previous context (EOS tokens at the start of a sequence), and the EOS rule is
  ``_shift_right_ignore_eos``. A token right after an EOS sees EOS as all its previous tokens.
* All arithmetic is exact integer arithmetic. The mix of every n-gram is below ``2^63``, so a 64-bit
  implementation is exact; ``row_ids`` is below ``head_offsets[H - 1] + head_vocab_sizes[H - 1]``.
* ``new_history`` holds the last ``ngram_size - 1`` tokens of ``history`` followed by ``token_ids``,
  the ``history`` of the next call of the same sequences. A stateful model keeps it as a state.
* The graph replaces the ids of padded positions by ``eos_token_id`` before the operation, as the
  reference model does with ``ple_input_ids``.
* The constants of Qwen3.8-Flash-Next are computed once from the configuration, or read from the
  checkpoint buffers ``layer_multipliers``, ``ngram_heads_vocab_sizes`` and ``ngram_heads_offsets``:
  ``ngram_size = 3``, ``heads_per_ngram = 8``, ``multipliers = [23703573157769, 20109073645365,
  8052911324071]``, ``head_vocab_sizes`` the 16 primes after 20,000,000 (20000003 ... 20000171),
  ``head_offsets`` their running sums (0, 20000003, ..., 300001275), ``eos_token_id = 248044``.
* The cost is ``O(N * L * (ngram_size + H))`` integer operations without memory traffic; the
  following *Gather* of ``H`` rows per token dominates.


**Attributes**

* *ngram_size*

  * **Description**: the largest n-gram order. Orders ``2 .. ngram_size`` are hashed.
  * **Range of values**: an integer ``>= 2``
  * **Type**: ``int``
  * **Required**: *yes*

* *heads_per_ngram*

  * **Description**: the number of hashed heads of every n-gram order.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *multipliers*

  * **Description**: the ``ngram_size`` odd hash multipliers of the token, the previous token, and
    so on. Every product of a token id and a multiplier must be below ``2^63``.
  * **Range of values**: positive integers
  * **Type**: ``int64[]``
  * **Required**: *yes*

* *head_vocab_sizes*

  * **Description**: the vocabulary size of every hashed head, ``H`` values.
  * **Range of values**: positive integers
  * **Type**: ``int64[]``
  * **Required**: *yes*

* *head_offsets*

  * **Description**: the first row of every hashed head in the packed table, ``H`` values. The rows
    of the packed table, ``head_offsets[h] + head_vocab_sizes[h]``, must be below ``2^31``, as
    ``row_ids`` is ``int32``; Qwen3.8-Flash-Next has about ``3.2 * 10^8`` rows.
  * **Range of values**: non-negative integers
  * **Type**: ``int64[]``
  * **Required**: *yes*

* *eos_token_id*

  * **Description**: the token that ends a document. Previous tokens before the start of the
    sequence or before an EOS are replaced by it.
  * **Range of values**: a non-negative integer
  * **Type**: ``int``
  * **Required**: *yes*


**Inputs**

* **1**: ``token_ids`` - 2D tensor of type *T_IND* and shape ``[N, L]``: the token ids of the current
  call. **Required.**

* **2**: ``history`` - 2D tensor of type *T_IND* and shape ``[N, ngram_size - 1]``: the previous
  ``ngram_size - 1`` tokens of every sequence, oldest first; ``eos_token_id`` for a new sequence.
  **Required.**


**Outputs**

* **1**: ``row_ids`` - 3D tensor of type ``int32`` and shape ``[N, L, H]``: the row of the packed
  n-gram table for every token and hashed head.

* **2**: ``new_history`` - 2D tensor of type *T_IND*, the type of ``history``, and shape
  ``[N, ngram_size - 1]``: the last ``ngram_size - 1`` tokens, the ``history`` of the next call.


**Types**

* *T_IND*: ``int32`` or ``int64``.


**Dimensions**

* ``N`` - batch size.

* ``L`` - number of tokens of the current call.

* ``H`` - number of hashed heads, ``(ngram_size - 1) * heads_per_ngram``.


**Example**

*Qwen3.8-Flash-Next prefill of 300 tokens*

.. code-block:: xml
   :force:

    <layer id="1" name="ple_hash" type="QSAHashFunction" version="extension">
        <data ngram_size="3" heads_per_ngram="8"
              multipliers="23703573157769,20109073645365,8052911324071"
              head_vocab_sizes="20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171"
              head_offsets="0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275"
              eos_token_id="248044"/>
        <input>
            <port id="0" precision="I64"> <!-- token_ids -->
                <dim>1</dim>    <!-- N -->
                <dim>300</dim>  <!-- L -->
            </port>
            <port id="1" precision="I64"> <!-- history: [248044, 248044] for a new sequence -->
                <dim>1</dim>    <!-- N -->
                <dim>2</dim>    <!-- ngram_size - 1 -->
            </port>
        </input>
        <output>
            <port id="2" precision="I32"> <!-- row_ids -> Gather from the 320001536 x 160 table -->
                <dim>1</dim>    <!-- N -->
                <dim>300</dim>  <!-- L -->
                <dim>16</dim>   <!-- H -->
            </port>
            <port id="3" precision="I64"> <!-- new_history: the type of history -->
                <dim>1</dim>    <!-- N -->
                <dim>2</dim>    <!-- ngram_size - 1 -->
            </port>
        </output>
    </layer>
