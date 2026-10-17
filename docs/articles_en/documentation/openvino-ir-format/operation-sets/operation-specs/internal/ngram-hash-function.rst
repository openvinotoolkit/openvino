NGramHashFunction
=================


.. meta::
  :description: Learn about NGramHashFunction - computes the row indices of the hashed n-gram
                embedding table (PLE) of Qwen3.8-Flash-Next from prepared n-grams, with exact 64-bit
                integer hashing inside the operation.

**Versioned name**: *NGramHashFunction*

**Category**: *Sequence processing*

**Short description**: *NGramHashFunction* hashes the bigrams, trigrams and longer n-grams that end at
every token into the row indices of a packed n-gram embedding table. It takes the n-gram of every token
already prepared, so it is stateless. The 64-bit integer arithmetic of the hash stays inside the operation; the inputs and outputs fit 32-bit integers.

**Detailed description**:

Qwen3.8-Flash-Next (``qwen4_exp`` in Hugging Face Transformers) adds a Per-Layer Embedding (PLE) at
decoder layer 2: an n-gram embedding looked up by hashes of the last tokens
(``Qwen4ExpTextNGramEmbedding``). The table has ``H = (ngram_size - 1) * heads_per_ngram`` hashed
heads, 16 for the model: heads 0-7 hash the bigram and heads 8-15 the trigram that ends at a token.
Every head has its own prime vocabulary size and its rows are packed into one table at an offset.

The input holds, for every token, its n-gram window of ``ngram_size`` token ids, oldest first:
``ngrams[..., ngram_size - 1]`` is the token itself and ``ngrams[..., ngram_size - 1 - k]`` is the token
``k`` steps back. The producer of the windows applies the document boundaries: a token before the start
of the sequence, or before an EOS token, is replaced by EOS, so that the n-grams of packed documents
never cross an EOS. *NGramHashFunction* computes, for every token and head, the row of the table:

* the n-gram of order ``o`` is mixed as ``(x[0] * m[0]) XOR (x[1] * m[1]) XOR ... XOR (x[o-1] * m[o-1])``,
  where ``x[k]`` is the token ``k`` steps back and ``m`` are the odd multipliers;
* head ``h`` of order ``o`` takes the row ``(mix_o mod head_vocab_sizes[h]) + head_offsets[h]``.

The products of a token id and a multiplier need up to 63 bits (multipliers
``< (2^63 - 1) / vocab_size``), and the modulo is taken of the 63-bit mix. The result rows, below the
table size of about ``3.2e8`` for the model, fit ``int32``. A device without native 64-bit integer
arithmetic in the graph (for example a plugin that converts ``i64`` to ``i32``) cannot evaluate the
hash with regular operations; *NGramHashFunction* keeps the 64-bit arithmetic inside its kernel, and
the constants as ``int64`` attributes.

The windows are prepared in the graph with regular operations: the history of the previous
``ngram_size - 1`` tokens of every sequence (an input of the model) is concatenated with the tokens of
the call, the window of every token is sliced out and the tokens across an EOS are replaced by EOS (see
``make_ngrams`` below); the last ``ngram_size - 1`` tokens are the history of the next call. The n-gram
embedding model of Qwen3.8-Flash-Next is the same with SDPA and with paged attention: it is not
transformed to paged attention, and the host keeps the history of every sequence.

The graph flow of the PLE lookup is *NGramHashFunction* -> *Gather* (``table``, ``row_ids``, axis 0)
-> *Reshape* ``[..., H * head_dim]``.

*NGramHashFunction* provides functionality according to the following pseudo-code using ``numpy``:

.. code-block:: py
   :force:

    def NGramHashFunction(ngrams, *, ngram_size, heads_per_ngram, multipliers, head_vocab_sizes, head_offsets):
        assert ngrams.shape[-1] == ngram_size
        x = ngrams.astype(numpy.int64)                          # exact integer math, every mix < 2**63
        mix = x[..., ngram_size - 1] * int(multipliers[0])      # the token itself
        rows = []
        for o in range(2, ngram_size + 1):                      # bigram, trigram, ...
            mix = mix ^ (x[..., ngram_size - o] * int(multipliers[o - 1]))   # the token o - 1 steps back
            for e in range(heads_per_ngram):
                h = (o - 2) * heads_per_ngram + e
                rows.append(mix % int(head_vocab_sizes[h]) + int(head_offsets[h]))
        return numpy.stack(rows, axis=-1).astype(numpy.int32)   # [..., H]

The windows are prepared by regular operations equivalent to:

.. code-block:: py
   :force:

    def make_ngrams(token_ids, history, *, ngram_size, eos_token_id):
        # token_ids [N, L], history [N, ngram_size - 1] (EOS for a new sequence), both of the same integer type
        c, L = ngram_size - 1, token_ids.shape[1]
        sequence = numpy.concatenate([history, token_ids], axis=1)                 # [N, c + L]
        back = [sequence[:, c - k:c - k + L] for k in range(c + 1)]                # back[k]: the token k steps back
        valid = numpy.ones_like(back[0], dtype=bool)
        window = [back[0]]
        for k in range(1, c + 1):          # back(k) is EOS if an EOS lies strictly between it and the token
            window.append(numpy.where(valid, back[k], eos_token_id))
            valid &= back[k] != eos_token_id
        ngrams = numpy.stack(window[::-1], axis=-1)                               # [N, L, ngram_size], oldest first
        return ngrams, sequence[:, -c:]                                            # windows, next history

Properties that follow from the definition:

* With windows prepared by ``make_ngrams``, the result
  equals the row indices of ``Qwen4ExpTextNGramEmbedding.forward``: the history plays the role of the
  cached previous context (EOS tokens at the start of a sequence), and the EOS rule is
  ``_shift_right_ignore_eos``. A token right after an EOS sees EOS as all its previous tokens.
* The operation is element-wise over the leading dimensions: ``[N, L, ngram_size]`` gives
  ``[N, L, H]``, and any other rank of the leading dimensions is handled the same way. It has no state
  and no notion of padding; the rows of padded positions are computed and ignored.
* All arithmetic is exact integer arithmetic. The mix of every n-gram is below ``2^63``, so a 64-bit
  implementation is exact; ``row_ids`` is below ``head_offsets[H - 1] + head_vocab_sizes[H - 1]``.
* The graph replaces the ids of padded positions by ``eos_token_id`` before preparing the windows, as
  the reference model does with ``ple_input_ids``.
* The constants of Qwen3.8-Flash-Next are computed once from the configuration, or read from the
  checkpoint buffers ``layer_multipliers``, ``ngram_heads_vocab_sizes`` and ``ngram_heads_offsets``:
  ``ngram_size = 3``, ``heads_per_ngram = 8``, ``multipliers = [23703573157769, 20109073645365,
  8052911324071]``, ``head_vocab_sizes`` the 16 primes after 20,000,000 (20000003 ... 20000171),
  ``head_offsets`` their running sums (0, 20000003, ..., 300001275).
* The cost is ``O(T * (ngram_size + H))`` integer operations for ``T`` tokens without memory traffic;
  the following *Gather* of ``H`` rows per token dominates.


**Attributes**

* *ngram_size*

  * **Description**: the largest n-gram order and the size of the last dimension of ``ngrams``.
    Orders ``2 .. ngram_size`` are hashed.
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


**Inputs**

* **0**: ``ngrams`` - tensor of type *T_IND* and shape ``[..., ngram_size]``: the n-gram window of
  every token, oldest first, the token itself last; tokens before the start of the sequence or across
  an EOS are ``eos_token_id``. For example ``[N, L, ngram_size]``. **Required.**


**Outputs**

* **0**: ``row_ids`` - tensor of type ``int32`` and shape ``[..., H]``: the row of the packed n-gram
  table for every token and hashed head.


**Types**

* *T_IND*: ``int32`` or ``int64``.


**Dimensions**

* ``...`` - leading dimensions, the tokens, for example ``[N, L]``.

* ``H`` - number of hashed heads, ``(ngram_size - 1) * heads_per_ngram``.


**Example**

*Qwen3.8-Flash-Next prefill of 300 tokens*

.. code-block:: xml
   :force:

    <layer id="1" name="ple_hash" type="NGramHashFunction" version="extension">
        <data ngram_size="3" heads_per_ngram="8"
              multipliers="23703573157769,20109073645365,8052911324071"
              head_vocab_sizes="20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171"
              head_offsets="0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275"/>
        <input>
            <port id="0" precision="I64"> <!-- ngrams -->
                <dim>1</dim>    <!-- N -->
                <dim>300</dim>  <!-- L -->
                <dim>3</dim>    <!-- ngram_size -->
            </port>
        </input>
        <output>
            <port id="1" precision="I32"> <!-- row_ids -> Gather from the 320001536 x 160 table -->
                <dim>1</dim>    <!-- N -->
                <dim>300</dim>  <!-- L -->
                <dim>16</dim>   <!-- H -->
            </port>
        </output>
    </layer>
