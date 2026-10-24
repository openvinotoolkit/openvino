.. {#openvino_docs_ops_internal_QSAIndexer}

QSAIndexer
==========

.. meta::
  :description: Learn about QSAIndexer - the standard, stateful block-level indexer for Qwen Sparse Attention (QSA). It consumes projected raw query and key tokens, per-token RoPE cos/sin, plus a summary cache and the fixed-size raw keys and RoPE row of the incomplete block as explicit state inputs, and emits selections containing only complete blocks together with the updated state.

**Versioned name**: *QSAIndexer*

**Category**: *Internal*

**Short description**:
The *QSAIndexer* is the **model-specific plan producer** for Qwen Sparse Attention (QSA) in its
**standard, non-paged, stateful** form. It consumes externally projected indexer queries and keys, applies query
RMSNorm/RoPE (with caller-supplied per-token ``rope_cos``/``rope_sin``), stages the raw keys of the incomplete
block, generates summaries for completed blocks, and produces block selections. The state -- a summary cache and
the raw keys and RoPE row of the incomplete block -- is passed as explicit inputs and returned as updated outputs
(a pure Functional State Interface), so it can be bound to ``ReadValue``/``Assign``. Only ``block_index_keys_cache`` grows
with the sequence length (one entry per completed block); the incomplete-block state has a fixed size. This makes
it suitable for **fixed-batch**
SDPA pipelines that do not use a paged KV cache. For continuous batching over a PagedAttention page table, see the separate
:doc:`PagedQSAIndexer <paged-qsa-indexer>` operation.

**State contract (Functional State Interface)**

Unlike the paged variant, *QSAIndexer* reads **no page table** and owns no opaque runtime state. Its logical
state consists of three tensors that are passed as inputs and returned, updated, as outputs: ``block_index_keys_cache``
stores completed-block summaries, ``past_incomplete_raw_k`` stores the raw keys of the one incomplete block that
precedes this call, and ``past_incomplete_rope`` stores the RoPE coefficients of that block's first token. The
incomplete-block tensors have a fixed size independent of the sequence length; ``block_index_keys_cache`` dim 2 describes
allocated capacity. The ``key_length`` input describes how many valid tokens each row holds *after* the invocation
(the number valid before it is derived, see "Row lengths and left padding"). The operation is a pure function of
its inputs: it does not modify its input tensors and has no side effect. The caller carries the state across
invocations by feeding outputs ``block_index_keys_cache``, ``past_incomplete_raw_k`` and ``past_incomplete_rope`` back to the
same state inputs of the next invocation (for example through ``Assign``/``ReadValue``).

**Rollback.** The state supports rolling back only within the *current* indexer block, i.e. the incomplete block
returned by the previous invocation. If the caller discards the last ``d`` tokens of a row and the shortened
``key_length' = key_length - d`` still lies in the same block (``floor(key_length' / r) == floor(key_length / r)``),
it simply passes the returned state with ``key_length'`` to the next invocation: the staged slots
``[0, key_length' % r)`` and the first-token RoPE row are still valid, ``block_index_keys_cache`` is unchanged, and the slots
beyond ``key_length' % r`` are overwritten when the tokens are replayed. Rolling back across a block boundary
(discarding a token that completed a block, or any earlier block) is **not** supported by this state: the keys of
completed blocks are not retained and their summaries are not reverted. Such a rollback is outside the contract of
the operation; a caller that needs it must retain earlier state tensors itself or re-process the affected tokens.
Beam branching copies the state: the incomplete-block tensors are constant-size and the valid prefix of
``block_index_keys_cache`` is copied per branch. An implementation MAY alias an output with the corresponding input to avoid
a copy, see "Aliasing of state inputs and outputs".

**Detailed description**

The *QSAIndexer* consumes externally projected indexer queries and keys; projection weights are not inputs. It
applies query RMSNorm and partial RoPE, stages the unnormalized and unrotated keys of the incomplete block,
generates summaries for completed blocks, and performs block scoring and top-k selection. The summary chain is
*non-associative* and *incremental*:
:math:`\mathrm{RMSNorm}(\mathrm{Mean}(K)) \ne \mathrm{Mean}(\mathrm{RMSNorm}(K))`, and a block summary can only be
maintained incrementally as new tokens arrive.

**Head layout and the index-head grouping**

The indexer distinguishes two head dimensions that are **derived from the input shapes**, not from an attribute:

- ``index_query`` carries ``H_iq`` index-*query* heads in its penultimate dimension (``[B, L, H_iq, D_idx]``).
- ``raw_index_keys`` carries ``H_ik`` index-*key* heads in its penultimate dimension (``[B, L, H_ik, D_idx]``).

``H_ik`` is configurable and taken from the ``raw_index_keys`` shape, never hard-coded to ``1``. This lets a single block
summary key be stored per index-key head, so models that need more than one key projection head are expressible
without adding a special-case attribute. The score for selection head ``sh`` is a grouped, per-query-head ReLU
reduction. The grouping is fully determined by the head counts:

.. math::

    H_{iq} \bmod H_{ik} = 0, \quad G = H_{iq} / H_{ik}, \quad \mathrm{kv}(h) = \lfloor h / G \rfloor, \quad
    \mathrm{sel}(h) = \lfloor \mathrm{kv}(h) \cdot H_{sel} / H_{ik} \rfloor

    s_{b, sh} = \mathrm{scale} \cdot \sum_{\substack{h \in [0, H_{iq}) \\ \mathrm{sel}(h) = sh}}
        \mathrm{ReLU}\!\left( \langle q_{h}, \bar{k}_{b, \mathrm{kv}(h)} \rangle \right)

where:

- ``H_iq % H_ik == 0`` and :math:`G = H_{iq} / H_{ik}` is the number of index-*query* heads sharing one
  index-*key* head,
- :math:`\mathrm{kv}(h) = \lfloor h / G \rfloor` maps index-query head :math:`h` to its index-key head
  :math:`\mathrm{kv}(h) \in [0, H_{ik})`,
- :math:`H_{sel} \in \{1, H_{ik}\}` is the number of selection heads, and
- :math:`\mathrm{sel}(h) = \lfloor \mathrm{kv}(h) \cdot H_{sel} / H_{ik} \rfloor` maps the index-key head of
  query head :math:`h` to its selection head.

:math:`\bar{k}_{b, \mathrm{kv}(h)}` is the per-index-key-head summary key of block :math:`b`. The ReLU is applied
per index-query head before reduction over heads, as specified by the QSA scoring semantics. Under QSA
(:math:`H_{iq} = 4`, :math:`H_{ik} = 1`, :math:`H_{sel} = 1`) all four index-query heads map to the single
index-key head and the single selection head, so the score is the sum over all four per-head ReLU dot-products.

**State organization**

The logical state is carried by three tensors. Let :math:`S_{cap}` be the token capacity of ``block_index_keys_cache`` (it
holds ``floor(S_cap / r)`` block entries) and :math:`r` be ``key_block_size``:

- ``block_index_keys_cache`` ``[B, H_ik, floor(S_cap / r), D_idx]`` stores mean-pooled, RMSNorm'ed, first-token-RoPE'ed
  summaries for complete blocks. For batch element ``b``, entries ``[0, floor(past[b] / r))`` are valid
  before the invocation, where ``past[b]`` is defined below. The output copy carries the valid entries
  ``[0, floor(key_length[b] / r))``: old entries unchanged, newly completed summaries added; old valid entries are
  not recomputed during ordinary incremental execution. The capacity is uniform across rows, so rows shorter than
  the longest row leave their trailing entries unused.
- ``past_incomplete_raw_k`` ``[B, H_ik, r - 1, D_idx]`` holds the projected, unnormalized, unrotated keys of the
  incomplete block that precedes this call. For batch element ``b`` let ``a[b] = past[b] % r``; the key of logical
  position ``p`` (with ``p`` in the incomplete block, ``p < past[b]``) is stored in slot ``p % r``, so slots
  ``[0, a[b])`` are valid in the input. Slot ``r - 1`` can never be valid in the input, because the key that
  completes a block always arrives in the same call. Contents of other slots are unspecified, so a batch row that
  starts a new sequence (``past[b] == 0``) needs no reset. In the output, slots ``[0, m[b])`` with
  ``m[b] = key_length[b] % r`` hold the keys of the new incomplete block; the remaining slots are unspecified.
- ``past_incomplete_rope`` ``[B, 2 * indexer_rotary_dim]`` holds the RoPE coefficients of the first logical token
  of the incomplete block, interleaving its ``rope_cos`` and ``rope_sin`` rows as ``(cos_0, sin_0, cos_1, sin_1, ...)``.
  It is valid in the input iff
  ``a[b] > 0`` and in the output iff ``m[b] > 0``; otherwise its content is unspecified.

When ``key_block_size == 1`` every token completes a block immediately; ``past_incomplete_raw_k`` then has shape
``[B, H_ik, 0, D_idx]`` and ``past_incomplete_rope`` is never read.

The raw-key history is not retained: the keys of a completed block are consumed once, when its summary is
computed, and are never read again. Scoring and top-k read only ``block_index_keys_cache``.

**Row lengths and left padding.** ``index_query``/``raw_index_keys``/``rope_cos``/``rope_sin`` carry ``L`` rows per batch
element. The ``key_length`` input gives the total number of valid tokens of each batch element *after* this
invocation, and replaces an explicit past length. The operation derives, per batch element ``b``:

.. math::

    n_{new}[b] = \min(L, \mathrm{key\_length}[b]), \quad \mathrm{pad}[b] = L - n_{new}[b], \quad
    \mathrm{past}[b] = \mathrm{key\_length}[b] - n_{new}[b]

If ``key_length[b] >= L``, all ``L`` rows are real tokens, there is no padding and ``past[b] = key_length[b] - L``
(decode and continuation chunks). If ``key_length[b] < L``, the row is a fresh sequence (``past[b] == 0``) whose
first ``pad[b] = L - key_length[b]`` rows are **left-padding**; the real tokens occupy rows ``[pad[b], L)``.
Left-padding of a row that has a non-empty past is not representable: such rows must be issued in a separate
invocation without padding, or use :doc:`PagedQSAIndexer <paged-qsa-indexer>`. Rows may differ in ``key_length``
and in padding within one invocation; all batch elements share the same ``L``. The caller MUST ensure
``key_length[b] <= S_cap``; the operation does not resize ``block_index_keys_cache``.

The **logical position** of the real token in row ``l >= pad[b]`` of batch element ``b`` is
``pos = key_length[b] - L + l`` (equivalently ``past[b] + l - pad[b]``). Padding rows are not tokens: they are not
stored, not pooled, occupy no logical position, and never contribute to any block. Logical positions therefore
start at ``0`` at the first real token, i.e. **block membership is defined in padding-stripped coordinates**. A
consumer whose dense key/value tensors keep the left-padding rows (such as :doc:`SparseSDPA <sparse-sdpa>`) must
translate logical positions by the per-row padding offset; see its ``key_length`` input. The logical position
drives block addressing and causal filtering only, never a RoPE angle. Query token ``(b, l)`` is rotated with
``(rope_cos[b, l], rope_sin[b, l])``; the summary of block ``j`` is rotated with the coefficients of the token at
logical position ``j * r`` (its first token): the row of this invocation if that token is among the current real
tokens, otherwise the row in ``past_incomplete_rope``. ``rope_cos``/``rope_sin`` always have a full ``[B, L, ...]``
shape (no broadcast over ``B``), so per-row position schemes and padding are expressible. Because the per-token
cos/sin are supplied by the producer (normally the same tensors the main attention RoPE consumes), any position
scheme -- mRoPE sections, non-monotonic or gapped positions, scaled RoPE variants -- is reproduced exactly without
an op change. A block is
*completed* once exactly :math:`r` raw keys have been consumed. The last partially-filled block is not
summarized and is never included in this operation's selection. The paired sparse-attention consumer derives the
causal tail directly from each query's logical position and processes its visible token prefix separately.

**Aliasing of state inputs and outputs.** An implementation MAY write an output state tensor into the storage of
the corresponding input (``block_index_keys_cache``, ``past_incomplete_raw_k``, ``past_incomplete_rope``) to avoid a copy,
provided the result is identical to the non-aliased one. In that case all reads of the pre-call content of
``past_incomplete_raw_k`` and ``past_incomplete_rope`` (by every batch row and every block of this call) MUST take
effect before any write of the post-call content to the same storage. A hazard exists whenever
``past[b] % r > 0``, ``key_length[b] % r > 0`` and the call completes at least one block: the slots read for the
old incomplete block overlap the slots written for the new one. An implementation MAY satisfy this with a separate
commit stage, double buffering, or any other mechanism that yields the same result as reading the old state into a
temporary first.

**Summary capacity.** The caller MUST ensure ``floor(S_cap / r) >= max_b floor(key_length[b] / r)`` before
invocation. The operation never grows ``block_index_keys_cache``; the wrapper that owns the state grows it (preserving the
valid entries ``[0, floor(past[b] / r))`` of every row) before the call, ideally with headroom so growth is
amortized rather than performed on every decode step. The output ``block_index_keys_cache`` has the same shape as the input.

**Score computation**

For a query at logical position :math:`p` with projected query :math:`q_h \in \mathbb{R}^{D_{idx}}` (per index-query
head :math:`h`) and block :math:`b` whose per-index-key-head summary key is
:math:`\bar{k}_{b, \mathrm{kv}(h)} \in \mathbb{R}^{D_{idx}}`, the per-selection-head score is the grouped reduction
over index-query heads (see the formula above), mapping each query head to its selection head via
:math:`\mathrm{sel}(h)`. The ReLU is applied *per index-query head before reduction over heads*, matching the
QSA scoring semantics. The indexer query :math:`q` arrives projected but not normalized
or rotated. The op applies query RMSNorm (via ``q_norm_weight``) and partial RoPE using the token's own
``(rope_cos, rope_sin)`` row. The pooled block key undergoes RMSNorm (via ``k_norm_weight``) and partial RoPE
using its first token's row (see "Row lengths and left padding"). The rotation is the split-half (rotate-half) RoPE
of Hugging Face Transformers over the first :math:`R = \mathrm{indexer\_rotary\_dim}` channels, computed in
``fp32``; the channels after :math:`R` are not rotated:

.. math::

    x' = x \odot \cos + \mathrm{rotate\_half}(x) \odot \sin, \quad
    \mathrm{rotate\_half}(x) = (-x_{R/2}, \ldots, -x_{R-1}, x_0, \ldots, x_{R/2-1})

with ``cos`` and ``sin`` the ``R``-wide ``rope_cos``/``rope_sin`` row of the token. In Hugging Face RoPE both halves
of a row are equal, so this is :math:`x'_j = x_j \cos_j - x_{j + R/2} \sin_j`,
:math:`x'_{j + R/2} = x_{j + R/2} \cos_j + x_j \sin_j` for :math:`0 \le j < R/2`.

The RMSNorm used for the pooled block key is defined as

.. math::

    \mathrm{rms\_norm}(x, w, \mathrm{epsilon}) = x \cdot \mathrm{rsqrt}\!\left( \mathrm{mean}(x^2) + \mathrm{epsilon} \right) \cdot w

computed in ``fp32`` regardless of the input precision. Qwen's Gemma-style scale :math:`(1 + w)` is folded by the
model converter into the stored weight :math:`w' = 1 + w`, so the op consumes the already-shifted weight.

**Numerics of the block summary.** The pooled key is accumulated in ``fp32`` and rounded to *T* before RMSNorm,
matching the reference model (``mean(k.float()).to(k.dtype)``); RMSNorm and RoPE are then evaluated in ``fp32``
and the result is stored to ``block_index_keys_cache`` as *T*. An implementation MAY keep the pooled mean in ``fp32``
instead of rounding it to *T*; the summaries then differ by at most one rounding of *T*, which can change the
selection only among blocks whose scores are equal within that error.

**Score scale.** ``scale`` multiplies every score of a query by the same positive constant, so it never changes
the ranking: in exact arithmetic ``selected_block_indices`` and ``selected_block_count`` do not depend on its value. It is kept for
parity with the reference scoring (``1 / sqrt(indexer_head_dim)``); an implementation MAY omit the
multiplication.

**Selection, tie-breaking, and top-k**

- A block is *causally valid* for a query at logical position :math:`p` iff its last token is at most :math:`p`,
  i.e. :math:`N_{complete}(p) = \lfloor (p + 1) / r \rfloor` blocks are eligible. Only these causally-valid
  complete blocks participate in the top-k (future / not-yet-written blocks never leak into the selection).
- The number of selected *complete* blocks per selection head is capped at ``block_topk``; the output width is
  ``K_max = block_topk``. The selection contains no causal-tail block ID, and ``selected_block_count`` counts complete
  selected blocks only: ``0 <= selected_block_count <= min(block_topk, n_complete)``.
- **Tie-breaking and ordering**: among equal scores the *smaller block index* (earlier block) wins; the op
  computes the selection as ``argsort(-scores, kind='stable')``, takes the first ``k`` entries, and emits them in
  **strictly ascending** block order (``sort(order[:k])``), yielding a deterministic, implementation-independent,
  monotone result. The valid output prefix ``selected_block_indices[t, sh, :selected_block_count[t, sh]]`` is sorted ascending.

**Block grid and causal rules**

The KV history of a batch element is partitioned into blocks of :math:`r` tokens. Block :math:`b` covers tokens
:math:`[b \cdot r, (b+1) \cdot r - 1]`. Scoring is causal: a query at logical position :math:`p` may only select
complete blocks with :math:`b < \lfloor (p+1)/r \rfloor`. The output contains only these selected complete-block
IDs; neither a tail ID nor tail token indices are emitted. For causal attention, the consumer independently
computes :math:`tail\_start = \lfloor (p+1)/r \rfloor r` and attends token positions
:math:`tail\_start, \ldots, p` (inclusive). This tail is empty when :math:`(p+1) \bmod r = 0`. The consumer
applies this rule per query and also enforces the token-level causal bound :math:`kpos \le q_{abs}`.

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def apply_partial_rope(x, cos, sin, rotary_dim):
      # Hugging Face rotate-half over the first rotary_dim channels, in fp32; cos/sin: [..., rotary_dim]
      half = rotary_dim // 2
      x = x.astype(float32)
      r = x[..., :rotary_dim].copy()
      rotated = concatenate([-r[..., half:], r[..., :half]], axis=-1)
      x[..., :rotary_dim] = r * cos + rotated * sin
      return x

    def qsa_indexer(q_raw, k_raw, rope_cos, rope_sin,
            key_length, q_norm_weight, k_norm_weight,
            block_index_keys_cache, past_incomplete_raw_k, past_incomplete_rope,
            *, key_block_size, block_topk, num_selection_heads, indexer_rotary_dim,
            indexer_head_dim, epsilon, scale):
      # q_raw: [B, L, H_iq, D_idx], external projection output, not normed/RoPE'd
      # k_raw: [B, L, H_ik, D_idx], external projection output, unnormed/unrotated
      # rope_cos, rope_sin: [B, L, indexer_rotary_dim], per-token rotate-half coefficients
      # key_length: [B], total valid tokens after this invocation
      # block_index_keys_cache: [B, H_ik, floor(S_cap/r), D_idx], state input
      # past_incomplete_raw_k: [B, H_ik, r - 1, D_idx], slot (pos % r) of the incomplete block, state input
      # past_incomplete_rope: [B, 2 * indexer_rotary_dim], interleaved (cos_j, sin_j), state input
      # returns selected_block_indices, selected_block_count, and the updated block_index_keys_cache, past_incomplete_raw_k, past_incomplete_rope
      B, L, H_iq, Di = q_raw.shape
      Hik = k_raw.shape[2]
      Hsel, r = num_selection_heads, key_block_size
      Dr = indexer_rotary_dim
      Kmax = block_topk
      G = H_iq // Hik
      assert H_iq % Hik == 0 and Hsel in (1, Hik) and r >= 1
      assert past_incomplete_raw_k.shape == (B, Hik, r - 1, Di)
      assert past_incomplete_rope.shape == (B, 2 * Dr)
      assert key_length.shape == (B,)
      assert block_index_keys_cache.shape[2] >= max(key_length // r)   # caller-grown capacity

      q = rms_norm(q_raw, q_norm_weight, epsilon)
      q = apply_partial_rope(q, rope_cos[:, :, None, :], rope_sin[:, :, None, :], Dr)

      # Functional form: the outputs are new tensors; the inputs are left untouched.
      new_summary = block_index_keys_cache.copy()
      new_raw = past_incomplete_raw_k.copy()
      new_rope = past_incomplete_rope.copy()

      selected_block_indices = full((B, Hsel, L, Kmax), -1, dtype=int32)
      selected_block_count = zeros((B, Hsel, L), dtype=int32)
      for batch in range(B):
        KL = key_length[batch]
        n_new = min(L, KL)
        pad = L - n_new                    # left-padding rows; pad > 0 implies past == 0
        S0 = KL - n_new                    # past: valid tokens before this invocation
        a = S0 % r                         # valid staged keys, slots [0, a)
        first = S0 - a                     # logical position of the incomplete block's first token
        # Logical key sequence starting at `first`: staged keys followed by this call's real keys.
        seq_k = concat([past_incomplete_raw_k[batch, :, :a, :], k_raw[batch, pad:].swapaxes(0, 1)], axis=1)  # [Hik, a+n_new, Di]

        def block_start_row(p):            # interleaved (cos_j, sin_j) row of logical position p (p % r == 0)
          if p >= S0:
            l = pad + (p - S0)
            return stack([rope_cos[batch, l], rope_sin[batch, l]], axis=-1).reshape(-1)
          return past_incomplete_rope[batch]

        for b in range(S0 // r, KL // r):  # blocks completed by this invocation
          lo = b * r - first
          grp = seq_k[:, lo:lo + r, :]
          pooled = mean(grp.astype(float32), axis=1).astype(k_raw.dtype)   # [Hik, Di]
          normed = rms_norm(pooled, k_norm_weight, epsilon)
          row = block_start_row(b * r)
          new_summary[batch, :, b, :] = apply_partial_rope(normed, row[0::2], row[1::2], Dr)

        m = KL % r                         # keys of the new incomplete block
        if m > 0:
          start = KL - m
          new_raw[batch, :, :m, :] = seq_k[:, start - first:start - first + m, :]
          new_rope[batch, :] = block_start_row(start)

        for l in range(pad, L):            # padding rows keep selected_block_indices = -1, selected_block_count = 0
          pos = KL - L + l
          n_complete = (pos + 1) // r
          for sh in range(Hsel):
            scores = score_complete_blocks(q[batch, l], new_summary[batch],
                             n_complete, sh, G, scale)
            k = min(block_topk, n_complete)
            order = stable_argsort_descending(scores)  # equal scores: smaller block ID first
            picked = sort(order[:k])                   # output IDs are ascending
            selected_block_indices[batch, sh, l, :k] = picked
            selected_block_count[batch, sh, l] = k
      return selected_block_indices, selected_block_count, new_summary, new_raw, new_rope


**Attributes**

* *key_block_size*

  * **Description**: :math:`r`, the number of raw token keys pooled into one block key.
  * **Type**: ``int``
  * **Default value**: ``4``
  * **Required**: *no*
  * **Constraints**: ``key_block_size >= 1``.

* *block_topk*

  * **Description**: Maximum number of *complete* blocks selected per selection head and the output width
    ``K_max``. The causal tail is not part of this output and does not consume a selection slot.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``block_topk >= 1``.

* *num_selection_heads*

  * **Description**: Number of selection heads :math:`H_{sel}` in the emitted contract. Must be ``1`` or
    ``H_ik`` (the index-key head count taken from ``raw_index_keys``). Each index-query head :math:`h` is mapped to a
    selection head via :math:`\mathrm{sel}(h) = \lfloor \mathrm{kv}(h) \cdot H_{sel} / H_{ik} \rfloor`, where
    :math:`\mathrm{kv}(h) = \lfloor h / G \rfloor` and :math:`G = H_{iq} / H_{ik}`.
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*
  * **Constraints**: ``num_selection_heads >= 1``; ``num_selection_heads == 1`` (QSA default) or ``num_selection_heads == H_ik``.

* *indexer_rotary_dim*

  * **Description**: RoPE dimension used by the indexer (partial rotary). Only the first
    ``indexer_rotary_dim`` channels of each query head and block summary are rotated; the trailing
    ``indexer_head_dim - indexer_rotary_dim`` channels are passed through unrotated. ``rope_cos`` / ``rope_sin``
    carry ``indexer_rotary_dim`` coefficients per token; a ``past_incomplete_rope`` row carries
    ``2 * indexer_rotary_dim`` interleaved values ``(cos_0, sin_0, cos_1, sin_1, ...)``.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: even, ``indexer_rotary_dim <= indexer_head_dim``.

* *indexer_head_dim*

  * **Description**: Head dimension :math:`D_{idx}` of the indexer.
  * **Type**: ``int``
  * **Required**: *yes*

* *epsilon*

  * **Description**: Epsilon for RMSNorm.
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*

* *scale*

  * **Description**: Positive multiplier applied to every block score (see "Score scale"). It does not affect
    the selection outputs.
  * **Type**: ``float``
  * **Default value**: ``1.0 / sqrt(indexer_head_dim)``
  * **Required**: *no*
  * **Constraints**: ``scale > 0``.


**Inputs**

* **0**: ``index_query``
  A 4D tensor of type *T* with shape ``[B, L, H_iq, D_idx]``.
  Raw projected indexer queries before indexer RMSNorm and RoPE. The operation applies both internally. **Required.**

* **1**: ``raw_index_keys``
  A 4D tensor of type *T* with shape ``[B, L, H_ik, D_idx]``.
  Current-step projected indexer keys. The index-key head count ``H_ik`` is derived from this shape and may be
  greater than one. The real keys (rows ``[pad[b], L)``) are pooled, together with the staged keys of the
  incomplete block, into newly completed-block summaries, and the trailing keys of the new incomplete block are
  returned in ``past_incomplete_raw_k``. Left-padding rows (see ``key_length``) are ignored. **Required.**

* **2**: ``rope_cos``
  A 3D tensor of type *T* with shape ``[B, L, indexer_rotary_dim]``. ``rope_cos[b, l, :]`` is the rotate-half RoPE
  cosine of query token ``(b, l)`` for the first ``indexer_rotary_dim`` channels, in the Hugging Face layout
  (``cat(freqs, freqs)``, both halves equal). Intended to be the same per-token cosine tensor consumed by the main
  attention RoPE; the operation does not recompute angles, so producers should evaluate them in ``fp32`` before
  any cast to *T*. **Required.**

* **3**: ``rope_sin``
  A 3D tensor of type *T* with shape ``[B, L, indexer_rotary_dim]``; the sine counterpart of ``rope_cos``
  with the same layout. **Required.**

* **4**: ``key_length``
  A 1D integer tensor with shape ``[B]``. For each batch element, the total number of valid tokens **after** this
  invocation (valid history plus the real tokens of this call). The operation derives
  ``n_new = min(L, key_length)``, ``pad = L - n_new`` and ``past = key_length - n_new``: ``key_length >= L`` means
  no padding and ``past = key_length - L``; ``key_length < L`` means a fresh sequence whose first ``L - key_length``
  rows are left-padding. Left-padding combined with a non-empty past is not representable (see "Row lengths and left
  padding"). This input is read-only; the operation does not modify it. The caller passes the post-call length, so no
  separate advance step exists. **Required.**

* **5**: ``q_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``. RMSNorm weight for raw projected indexer queries. **Required.**

* **6**: ``k_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``.
  RMSNorm weight for pooled block keys. **Required.**

* **7**: ``block_index_keys_cache``
  A 4D state tensor of type *T* with shape ``[B, H_ik, floor(S_cap / key_block_size), D_idx]``. Completed-block
  summaries. Valid entries ``[0, floor(past[b] / r))`` of each row are read; the updated tensor is returned as
  output 2. The capacity MUST satisfy ``floor(S_cap / r) >= max_b floor(key_length[b] / r)``; growth is the caller's
  (wrapper's) responsibility and MUST preserve the valid entries of every row. **Required.**

* **8**: ``past_incomplete_raw_k``
  A 4D state tensor of type *T* with shape ``[B, H_ik, key_block_size - 1, D_idx]``. The projected, unnormalized,
  unrotated keys of the incomplete block that precedes this call: the key of logical position ``p`` lives in slot
  ``p % r``, and slots ``[0, past % r)`` are read. Fixed size, independent of the sequence length. The updated
  tensor is returned as output 3. **Required.**

* **9**: ``past_incomplete_rope``
  A 2D state tensor of type *T* with shape ``[B, 2 * indexer_rotary_dim]``. The RoPE coefficients of the first
  logical token of the incomplete block, interleaved as ``(cos_0, sin_0, cos_1, sin_1, ...)``: ``row[2j]`` is
  ``rope_cos[.., j]`` and ``row[2j + 1]`` is ``rope_sin[.., j]`` for ``0 <= j < indexer_rotary_dim``. Read iff
  ``past % r > 0``. Fixed size. The
  updated tensor is returned as output 4. **Required.**


**Outputs**

* **0**: ``selected_block_indices``
  A 4D tensor of type *T_IND* with shape ``[B, H_sel, L, K_max]``, where ``K_max = block_topk``.
  Contains only selected complete-block IDs. The valid prefix ``selected_block_indices[..., :selected_block_count]`` is **strictly
  ascending and duplicate-free**; ``-1`` is padded only after the prefix (no interior holes). Rows ``l < pad[b]``
  (left-padding) are all ``-1``. The causal tail is not represented in this tensor; SparseAttention derives it per
  query from the logical position. The standard form has no ``B_q``: one selection per query token, so the third
  dimension is ``L``. **Required.**

* **1**: ``selected_block_count``
  A 3D tensor of type *T_IND* with shape ``[B, H_sel, L]``.
  Number of selected complete-block IDs per selection head per query, satisfying
  ``0 <= selected_block_count <= min(block_topk, n_complete)``. Tail tokens are not counted. Left-padding rows have
  ``selected_block_count == 0``. **Required.**

* **2**: ``block_index_keys_cache``
  A 4D tensor of type *T* with the shape of input 7. Updated summary cache: valid entries
  ``[0, floor(key_length[b] / r))`` of each row. **Required.**

* **3**: ``past_incomplete_raw_k``
  A 4D tensor of type *T* with the shape of input 8. The keys of the new incomplete block in slots
  ``[0, key_length[b] % r)``; other slots are unspecified. **Required.**

* **4**: ``past_incomplete_rope``
  A 2D tensor of type *T* with the shape of input 9. The interleaved ``(cos_j, sin_j)`` RoPE row of the
  first token of the new incomplete block when ``key_length[b] % r > 0``; otherwise unspecified. **Required.**

Outputs 2-4 are the updated state and are meant to be fed back to inputs 7-9 of the next invocation.

For the QSA configuration ``H_sel = 1``, flattening the batch and query-token axes gives the selection ABI
``selected_block_indices[T, K]`` and ``selected_block_count[T]``, with ``K = block_topk``. The explicit ``B`` and ``H_sel`` axes above
are retained for the generalized batched/multi-selection-head form.

**State semantics**

* The operation is a pure function: it does not modify any input tensor and has no side effect, so it MAY be
  eliminated if its outputs are unused and MAY be reordered with respect to operations that do not consume its
  outputs. The state dependency between consecutive invocations is carried only by the tensors that connect output
  2-4 to inputs 7-9 (typically through ``Assign``/``ReadValue``).
* Before invocation, the first ``floor(past[b] / r)`` summary entries MUST be valid, the first ``past[b] % r`` slots
  of ``past_incomplete_raw_k`` MUST hold the keys of the incomplete block, and ``past_incomplete_rope`` MUST hold
  its first-token row when ``past[b] % r > 0``.
* Summary capacity growth and beam branching (by copying the state) are managed by the caller. Rollback is limited
  to the current indexer block (see "Rollback"). Distinct active branches MUST NOT share a state tensor that is
  updated in place.
* An implementation MAY alias outputs 2-4 with inputs 7-9 only if it preserves the read-before-write ordering
  described in "Aliasing of state inputs and outputs" and the input state is not used again by the caller;
  otherwise it MUST allocate separate output storage.

**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` (``key_length`` included).
* ``index_query`` and ``raw_index_keys`` share the first two dims ``[B, L]``; both last dimensions equal ``indexer_head_dim``.
  ``index_query`` penultimate dimension is ``H_iq`` and ``raw_index_keys`` penultimate dimension is ``H_ik``, with
  ``H_iq % H_ik == 0`` and ``G = H_iq / H_ik`` index-query heads sharing each index-key head.
* ``H_ik`` (index-key heads) is taken from ``raw_index_keys`` shape; ``H_sel`` is ``1`` or ``H_ik``.
* ``block_index_keys_cache`` dim 0 equals ``B``, dim 1 equals ``H_ik``, dim 2 equals ``floor(S_cap / key_block_size)`` with
  ``floor(S_cap / key_block_size) >= max_b floor(key_length[b] / key_block_size)``, and dim 3 equals ``D_idx``.
* ``past_incomplete_raw_k`` has shape ``[B, H_ik, key_block_size - 1, D_idx]`` and ``past_incomplete_rope`` has
  shape ``[B, 2 * indexer_rotary_dim]``; both have dim 0 equal to ``B`` and a size independent of the sequence
  length.
* Outputs 2, 3 and 4 have exactly the shapes and element types of inputs 7, 8 and 9 respectively.
* ``key_length`` shape equals ``[B]`` with ``0 <= key_length[b] <= S_cap``. The derived ``n_new``, ``pad`` and
  ``past`` follow "Row lengths and left padding".
  ``block_index_keys_cache[b, :, :floor(past[b] / key_block_size), :]`` contains the valid completed summaries, and the
  first ``past[b] % key_block_size`` slots of ``past_incomplete_raw_k[b]`` the keys of the incomplete block.
* ``rope_cos`` and ``rope_sin`` both have shape ``[B, L, indexer_rotary_dim]`` (full ``B``, no broadcast),
  sharing ``[B, L]`` with ``index_query``. Rows ``l < pad[b]`` are left-padding and their content is ignored.
* ``selected_block_indices`` last dimension ``K_max`` equals ``block_topk``; ``selected_block_count[..., sh] <= K_max`` and is at most
  ``min(block_topk, n_complete)`` for each query. The
  valid prefix ``selected_block_indices[..., sh, :selected_block_count[..., sh]]`` is **strictly ascending** and contains no ``-1``
  entries.
* For query ``l >= pad[b]`` in batch element ``b``, the logical position is ``key_length[b] - L + l``. Only the
  incomplete block is carried in the state; the keys of completed blocks are not retained.
* If ``selected_block_count == 0``, a causal SparseAttention consumer may still produce a nonzero result from its
  automatically-derived tail. It produces zero only when no effective key remains after selection and tail
  handling.

**Types**

* *T*: any floating point type.
* *T_IND*: ``int32``.


**Example**

The example below shows one query token (``B=1``, ``L=1``) in a decode step with ``key_length[0] = 10``, i.e.
``n_new = 1``, ``pad = 0``, ``past = 9`` and logical position ``key_length[0] - L + l = 9``, with
``key_block_size = 4``, ``S_cap = 16`` and ``block_topk = 2`` (so ``K_max = 2``). ``block_index_keys_cache`` has capacity
``floor(16 / 4) = 4`` summary slots, of which the first two are valid (``4 >= floor(10 / 4) = 2``).
``past % r = 1``, so slot ``0`` of ``past_incomplete_raw_k`` holds the key of position ``8`` and
``past_incomplete_rope`` holds the RoPE row of that token. After the call ``key_length % r = 2``: the returned
``past_incomplete_raw_k`` holds the keys of positions ``8`` and ``9`` in slots ``[0, 2)``, the returned
``past_incomplete_rope`` still holds the row of position ``8``, and the returned ``block_index_keys_cache`` is unchanged
(no block was completed). The history covers complete blocks ``{0:[0..3], 1:[4..7]}`` and an incomplete block
``2:[8..9]``. Causally-valid complete
blocks for ``pos = 9`` are ``{0, 1}``. After a stable top-2 by score, the selection head emits
``selected_block_indices = [[[[0, 1]]]]`` and ``selected_block_count = [[[2]]]`` in strictly ascending order. The consumer independently
derives the tail as logical tokens ``[8, 9]`` for this query; neither its block ID nor its token positions appear in
the selection output. With ``H_iq = 4`` and ``H_ik = 1``, all four index-query heads participate in the single
selection head's score.

.. code-block:: xml
   :force:

   <layer ... type="QSAIndexer">
       <data block_topk="2" num_selection_heads="1" indexer_head_dim="128"
             indexer_rotary_dim="64" key_block_size="4" epsilon="1e-6"
             scale="0.088388"/>
       <input>
           <port id="0">   <!-- index_query: [B, L, H_iq, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="1">   <!-- raw_index_keys: [B, L, H_ik, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>1</dim><dim>128</dim>
           </port>
           <port id="2">   <!-- rope_cos: [B, L, rotary_dim] -->
               <dim>1</dim><dim>1</dim><dim>64</dim>
           </port>
           <port id="3">   <!-- rope_sin: [B, L, rotary_dim] -->
               <dim>1</dim><dim>1</dim><dim>64</dim>
           </port>
           <port id="4">   <!-- key_length: [B] -->
               <dim>1</dim>
           </port>
           <port id="5">   <!-- q_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
           <port id="6">   <!-- k_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
           <port id="7">   <!-- block_index_keys_cache: [B, H_ik, floor(S_cap/r), D_idx], state input -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="8">   <!-- past_incomplete_raw_k: [B, H_ik, r-1, D_idx], state input -->
               <dim>1</dim><dim>1</dim><dim>3</dim><dim>128</dim>
           </port>
           <port id="9">   <!-- past_incomplete_rope: [B, 2*rotary_dim], state input -->
               <dim>1</dim><dim>128</dim>
           </port>
       </input>
       <output>
           <port id="10" precision="I32">  <!-- selected_block_indices: [B, H_sel, L, K_max=block_topk] -->
               <dim>1</dim><dim>1</dim><dim>1</dim><dim>2</dim>
           </port>
           <port id="11" precision="I32">  <!-- selected_block_count: [B, H_sel, L] -->
               <dim>1</dim><dim>1</dim><dim>1</dim>
           </port>
           <port id="12">  <!-- block_index_keys_cache: updated state -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="13">  <!-- past_incomplete_raw_k: updated state -->
               <dim>1</dim><dim>1</dim><dim>3</dim><dim>128</dim>
           </port>
           <port id="14">  <!-- past_incomplete_rope: updated state -->
               <dim>1</dim><dim>128</dim>
           </port>
       </output>
   </layer>