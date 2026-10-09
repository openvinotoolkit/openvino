.. {#openvino_docs_ops_internal_PagedQSAIndexer}

PagedQSAIndexer
===============

.. meta::
  :description: Learn about PagedQSAIndexer - a paged, model-specific plan producer for Qwen Sparse Attention (QSA) under continuous batching / PagedAttention page tables. It maintains an incremental pooled-key summary cache over the main page table plus a per-slot pending ring, supports Prefix Caching, and emits a clean (sel_indices, sel_count) block-selection contract for the unified SparseSDPA / SparsePA consumers.

**Versioned name**: *PagedQSAIndexer*

**Category**: *Internal*

**Short description**:
The *PagedQSAIndexer* is the **model-specific plan producer** for Qwen Sparse Attention (QSA) in its
**paged** form, used under **continuous batching / PagedAttention page tables**. Unlike the standard
:doc:`QSAIndexer <qsa-indexer>` (a pure Functional State Interface with external ``ReadValue``/``Assign`` state),
*PagedQSAIndexer* owns its recurrent caches **in place** as explicitly-passed paged tensors and reads the shared
main PagedAttention page table (``block_indices`` / ``block_indices_begins``). It performs the same *K-side
compression chain* (per-token or pooled key mean-pool, RMSNorm, block-start RoPE), maintains the incremental
``summary_cache`` and the per-sequence ``indexer_k_cache`` ring, and emits a clean ``(sel_indices, sel_count)``
block-selection contract. It **supports page-aligned prefix caching** (see :ref:`Prefix Caching` below): because
``page_size % compress_ratio == 0``, cached physical pages already have their ``summary_cache`` populated, so a
matched prompt prefix skips summary recomputation for the cached pages.

**Data flow and cache organization**

The K-side history is maintained as two in-place paged caches plus one small per-slot RoPE-start buffer. These
reuse the main PagedAttention page table and are passed as explicit inputs so a unified scheduler can manage
their tier residency:

- ``summary_cache`` ``[num_blocks, H_ik, page_size / compress_ratio, D_idx]``: the mean-pooled, RMSNorm'ed,
  block-start-RoPE'ed summary key of each *completed* block. This is the incrementally-updated artifact that
  makes the per-token update :math:`O(1)`. It is addressed by the **same page table** as the main PagedAttention
  KV cache (``block_indices`` / ``block_indices_begins``). For sequence ``s``, block ``b`` (covering logical
  tokens ``[b*r, (b+1)*r - 1]``) is stored at the physical location
  ``summary_cache[phys, :, slot_in_block, :]`` with
  ``phys = block_indices[block_indices_begins[s] + (b * r) // page_size]`` and
  ``slot_in_block = (b * r % page_size) // r``. The second dimension is the **configurable index-key head count
  ``H_ik``** (never hard-coded to ``1``). This requires
  ``page_size % compress_ratio == 0`` so every summary slot maps to a fixed position inside one physical page.

- ``indexer_k_cache`` ``[num_slots, H_ik, ring_capacity, D_idx]``: the projected, per-token indexer keys of the
  *currently incomplete* block(s) of each sequence, held in a **per-sequence ring buffer** (addressed by
  ``slot_mapping``). The second dimension is the **configurable index-key head count ``H_ik``**. The ring capacity
  ``C`` is a **real functional parameter**: ``C % compress_ratio == 0`` and ``C >= compress_ratio``. A token at
  logical position :math:`pos` is stored at ring offset :math:`off = pos \bmod C` in
  ``indexer_k_cache[slot, :, off, :]``. Position addressing makes the op **idempotent and decode-continuous**:
  re-running a step writes the same slot (overwriting stale content), and a chunked prefill that resumes at
  ``off == 0`` re-establishes the same block-start state without any internal counter.
- ``indexer_k_start_cos_sin`` ``[num_slots, C/r, 2 * indexer_rotary_dim]``: the precomputed RoPE ``(cos, sin)`` at
  the block-start position of each possible pending block start within the ring, addressed by
  ``(pos // compress_ratio) % (C / compress_ratio)``. It is refreshed when the first token of a block arrives.
- ``k_rope_cos_sin`` ``[tokens, 2 * indexer_rotary_dim]``: key-side RoPE ``(cos, sin)``, used **only to capture
  the block-start phase** (decouples logical index from RoPE coordinate).

A block is *completed* once exactly :math:`r` raw keys have been appended (the pending ring fills and its summary
is flushed into ``summary_cache``). The flush pools ``indexer_k_cache[slot, :, (b*r) % C : (b*r) % C + r, :]`` —
the ring slice holding exactly the ``r`` keys of block :math:`b`. The last, partially-filled block of a sequence
is *not* summarized into ``summary_cache``; instead it is emitted as a **valid block index** in the output (the
producer-completeness rule), so the consumer gathers it as a partial block without any ``r-1`` tail hack.

.. _Prefix Caching:

**Prefix Caching**

Because ``page_size % compress_ratio == 0``, a cached physical page always has every ``summary_cache`` slot for
its completed blocks already populated — the summary layout is page-aligned. Prefix caching is **page-aligned**:
the scheduler only matches prefixes whose length is a whole multiple of ``page_size``.

.. note::
   **Why page-aligned matching is required.** Since ``page_size % compress_ratio == 0``, a matched prefix of any
   ``matched_len`` satisfies ``matched_len % compress_ratio == 0``, so the new sequence always starts a fresh
   block at logical position ``past_lens[B] = matched_len`` with ``P = 0`` pending keys. Its own ring slot is
   freshly written: the first suffix token lands at ``off == 0`` and overwrites the block-start phase and the ring
   contents, so the old content of that sequence's ring slot is irrelevant. The shared pages' summary rows are only
   **read**; the suffix writes into **new** pages, so no copy-on-write is needed. If the matched prefix is not a
   whole multiple of ``page_size``, it must be **rounded down** to the page boundary (``matched_len := matched_len -
   matched_len % page_size``); otherwise the shared page's summary rows would not correspond to a fully-written
   block set and the pool would read stale raw keys.

**Execution flow.** When a new request's prompt shares a page-aligned prefix with a previously-cached one, the
scheduler:

1. Detects the shared prefix length, **rounds it down to a multiple of ``page_size``**, and advances ``past_lens``
   by that length (``past_lens[B] = matched_len``). The shared KV pages — and therefore their ``summary_cache``
   rows — are reused by the page table; ``block_indices[B]`` points at the shared pages first, then at new pages
   for the suffix.
2. The op sees ``P = 0`` pending keys (the prefix is block-complete). It processes only the *unmatched suffix*
   (logical positions ``>= past_lens[B]``), writing raw keys into its own ring slot and flushing completed blocks
   into the **new** suffix pages. No COW, no overwrite of the shared summary rows.
3. The op **skips recomputing summaries for the cached prefix pages**: it only scores against the existing summary
   keys of the shared prefix blocks (reading them directly) plus the newly-written suffix summaries.

This is *lossless* with respect to the sparse plan **given a deterministic compression kernel**: the summary keys
for the prefix blocks are byte-identical to what a fresh prefill would have produced, so the emitted
``(sel_indices, sel_count)`` for the suffix is exactly what it would be had the prefix been re-computed. The cost
of prefix caching is therefore only the *scoring* over the (already-summarized) prefix blocks, not the
*construction* of those summaries.

**Worked example.** Let ``page_size = 16`` and ``compress_ratio = 4``. Request A writes 40 tokens:
page ``P0 = [0..15]`` (blocks 0–3), page ``P1 = [16..31]`` (blocks 4–7), page ``P2 = [32..39]`` (blocks 8–9, the
last page partially filled). Request B shares the first 35 tokens of A. Whole pages shared: ``32 = floor(35 / 16) *
16``, so ``past_lens[B] = 32`` and ``block_indices[B] = [P0, P1, Pnew]``. B's suffix is tokens ``32..`` in ``Pnew``;
it processes logical positions ``32, 33, ...`` (fresh ring writes, ``P = 0``). At logical position ``pos = 40``
(block ``10``), B scores blocks ``0..7`` from shared pages ``P0/P1`` and blocks ``8, 9`` from its own ``Pnew``
(just flushed).

**Counter-example (why page-aligned rounding is mandatory).** If one instead matched ``34`` tokens, tokens ``32, 33``
would be left in A's ring slot — they are *raw keys of A*, not of B. B's block ``8`` would then be pooled from
garbage (A's tokens), so prefix caching must round the match down to the page boundary.

.. note::
   **Decode mode and rollback.** With ``ring_capacity = C = compress_ratio`` (exactly one block per slot), the ring
   supports **standard greedy / autoregressive decoding only**. To support rollback of a speculative-decode draft
   of length :math:`N`, set :math:`C \ge r + N`; the scheduler then replays the draft by resending with a *smaller*
   ``past_lens``, and the op overwrites the ring slots **by position** (``off = pos % C``), so the stale draft
   tokens are simply overwritten by the replayed prefix — no explicit eviction is needed.

**Logical index vs. RoPE coordinate (M-RoPE)**

The op separates the *logical KV index* used for block addressing and causal filtering from the *RoPE coordinate*
used for positional rotation. The **logical position** of a token is derived from ``past_lens`` and the sequence
partition (``subsequence_begins``): for sequence ``s`` and token ``t``,
:math:`pos = past\_lens[s] + (t - subsequence\_begins[s])`. This logical position drives (a) which blocks are
causally valid and (b) the pending ring slot addressing. The **RoPE coordinate** for the key side is a purely
positional quantity, captured once per block in ``indexer_k_start_cos_sin``; the query arrives already RoPE'ed;
``k_rope_cos_sin`` supplies the key-side (possibly M-RoPE) ``(cos, sin)`` per token and is used only to capture
each block's start phase. This decoupling is what makes 3-axis M-RoPE expressible.

**Score computation**

For a query at logical position :math:`p` with projected query :math:`q_h \in \mathbb{R}^{D_{idx}}` (per index-query
head :math:`h`) and block :math:`b` whose per-index-key-head summary key is
:math:`\bar{k}_{b,\mathrm{kv}(h)} \in \mathbb{R}^{D_{idx}}`, the per-selection-head score is a **grouped reduction
over index-query heads**:

.. math::

    H_{iq} \bmod H_{ik} = 0, \quad G = H_{iq} / H_{ik}, \quad \mathrm{kv}(h) = \lfloor h / G \rfloor, \quad
    \mathrm{sel}(h) = \lfloor \mathrm{kv}(h) \cdot H_{sel} / H_{ik} \rfloor

    s_{b, sh} = \frac{1}{\sqrt{D_{idx}}} \sum_{\substack{h \in [0, H_{iq}) \\ \mathrm{sel}(h) = sh}}
        \mathrm{ReLU}\!\left( \langle q_{h},\, \bar{k}_{b,\mathrm{kv}(h)} \rangle \right)

where :math:`\bar{k}_{b,\mathrm{kv}(h)} =
\mathrm{RoPE}_{b\cdot r}(\mathrm{RMSNorm}(\tfrac{1}{r}\sum_{j=0}^{r-1} k_{b\cdot r + j,\mathrm{kv}(h)}))`
is the per-index-key-head block summary key. The ReLU is applied *per index-query head before reduction over
heads*, matching the reference QSA implementation. Under QSA (:math:`H_{iq} = 4`, :math:`H_{ik} = 1`,
:math:`H_{sel} = 1`) all four index-query heads map to the single selection head.

**Selection heads vs. index-key heads**

- ``H_ik`` (index-key heads): the heads of the indexer *key* projection, taken from the ``k_idx`` shape. Each
  produces an independent per-block summary key.
- ``H_iq`` (index-query heads): the heads of the indexer *query* projection, taken from the ``q_idx`` shape, with
  ``H_iq % H_ik == 0``.
- ``h_sel`` (:math:`H_{sel}`): the heads of the emitted *selection*, in ``{1, H_ik}``. Under QSA,
  :math:`H_{sel} = 1` — a single shared selection drives every query/KV head of the main attention.

The grouping is fully determined by ``H_iq % H_ik == 0`` and ``H_sel`` (see the formula above); there is no
additional partition constraint.

The RMSNorm used for the pooled block key is defined as

.. math::

    \mathrm{rms\_norm}(x, w, \mathrm{eps}) = x \cdot \mathrm{rsqrt}\!\left( \mathrm{mean}(x^2) + \mathrm{eps} \right) \cdot w

computed in ``fp32`` regardless of the input precision. Qwen's Gemma-style scale :math:`(1 + w)` is folded by the
model converter into the stored weight :math:`w' = 1 + w`, so the op consumes the already-shifted weight.

**Selection, tie-breaking, and top-k**

- A block is *causally valid* for a query at logical position :math:`p` iff its last token is at most :math:`p`,
  i.e. :math:`N_{complete}(p) = \lfloor (p + 1) / r \rfloor`. Only these causally-valid complete blocks participate
  in the top-k.
- The number of selected *complete* blocks per selection head is capped at ``block_topk``; the output width is
  ``K_max = block_topk + 1`` (the extra slot holds the causal-diagonal / incomplete block, when present).
- **Tie-breaking and ordering**: among equal scores the *smaller block index* (earlier block) wins; the op
  computes the selection as ``argsort(-scores, kind='stable')``, takes the first ``k`` entries, and emits them in
  **strictly ascending** block order (``sort(order[:k])``), yielding a deterministic, implementation-independent,
  monotone result. A plugin may rely on ``sel_indices[t, sh, :sel_count]`` being sorted ascending.
- **Causal-diagonal inclusion**: when the query at logical position :math:`p` lies inside an incomplete block
  (i.e. :math:`(p+1) \bmod r \neq 0`), the diagonal block with logical index :math:`\lfloor (p+1)/r \rfloor` is
  written **immediately after** the selected complete blocks, at ``sel_indices[..., k]``, and ``sel_count = k+1``.
  Since it is always greater than every selected complete block, the prefix stays strictly ascending.

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def paged_qsa_indexer(q_idx, k_idx, k_rope_cos_sin, summary_cache, indexer_k_cache,
                          indexer_k_start_cos_sin, k_norm_weight,
                          slot_mapping, block_indices, block_indices_begins,
                          subsequence_begins, past_lens,
                          *, compress_ratio, block_topk, h_sel,
                          indexer_rotary_dim, indexer_head_dim, page_size,
                          ring_capacity, num_sequences, eps, scale):
        # q_idx:            [tokens, H_iq * D_idx]   (projected, RMSNorm'ed and RoPE'ed externally)
        # k_idx:            [tokens, H_ik * D_idx]    (projected by external GEMM; H_ik configurable)
        # k_rope_cos_sin:   [tokens, 2 * indexer_rotary_dim]  (key-side RoPE (cos,sin); block-start phase)
        # summary_cache:    [num_blocks, H_ik, page_size // r, D_idx]
        # indexer_k_cache:  [num_slots, H_ik, ring_capacity, D_idx]  (per-sequence ring, capacity C)
        # indexer_k_start_cos_sin: [num_slots, C // r, 2 * indexer_rotary_dim]
        # k_norm_weight:    [D_idx]
        # slot_mapping:     [num_sequences]   (seq -> ring slot)
        # block_indices:    [total_logical_blocks]   logical -> physical block map (shared with main PA)
        # block_indices_begins: [num_sequences + 1]  split pointers into block_indices (len == total_logical_blocks)
        # subsequence_begins:   [num_sequences + 1]  split pointers into the token stream (len == tokens)
        # past_lens:        [num_sequences]  (logical length of each sequence before this step)
        # returns: sel_indices [tokens, H_sel, K_max]  (int32; -1 = pad), K_max = block_topk + 1
        #          sel_count   [tokens, H_sel]         (int32)
        tokens = q_idx.shape[0]
        Hiq = q_idx.shape[1] // indexer_head_dim      # index-query heads, derived from the shape
        Hik = k_idx.shape[1] // indexer_head_dim      # index-key heads, derived from the shape
        Hsel = h_sel
        r = compress_ratio
        C = ring_capacity
        Kmax = block_topk + 1
        G = Hiq // Hik                                # index-query heads per index-key head
        n_block_phases = C // r                       # ring holds C//r possible block-start phases
        assert page_size % r == 0 and Hiq % Hik == 0 and Hsel in (1, Hik) and C % r == 0 and C >= r

        k = reshape(k_idx, (tokens, Hik, indexer_head_dim))
        q = reshape(q_idx, (tokens, Hiq, indexer_head_dim))
        # 1. Per-token K-side update, addressed by logical position (idempotent / decode-continuous).
        #    off = pos % C selects the ring slot; the block-start phase is addressed by (pos//r) % (C//r);
        #    when pos % r == r-1 the block `pos // r` completes and is flushed via the shared page table.
        for s in range(num_sequences):
            slot = slot_mapping[s]
            base = past_lens[s]                       # logical position of the first token of this chunk
            for t in range(subsequence_begins[s], subsequence_begins[s + 1]):
                pos = base + (t - subsequence_begins[s])    # global logical position
                off = pos % C
                if pos % r == 0:
                    # first token of a block: record the block-start RoPE phase from this token's table row
                    indexer_k_start_cos_sin[slot, (pos // r) % n_block_phases] = k_rope_cos_sin[t]
                indexer_k_cache[slot, :, off, :] = k[t]      # [Hik, Di] (H_ik configurable)
                if pos % r == r - 1:
                    # block `b = pos // r` is now complete: mean-pool (fp32), RMSNorm, partial RoPE, flush.
                    # Its r keys live in the ring slice [(b*r) % C, (b*r) % C + r).
                    b = pos // r
                    ring_start = (b * r) % C
                    pooled = mean(indexer_k_cache[slot, :, ring_start:ring_start + r, :]
                                  .astype(float32), axis=1).astype(k.dtype)       # [Hik, Di]
                    nb = rms_norm(pooled, k_norm_weight, eps)                     # [Hik, Di]
                    bar = concat([rope(nb[..., :indexer_rotary_dim],
                                       indexer_k_start_cos_sin[slot, (b) % n_block_phases]),
                                  nb[..., indexer_rotary_dim:]], axis=-1)         # keep unrotated tail
                    phys = block_indices[block_indices_begins[s] + (b * r) // page_size]
                    slot_in_block = (b * r % page_size) // r
                    summary_cache[phys, :, slot_in_block, :] = bar                # [Hik, Di]

        # 2. Score the causally-valid complete blocks per selection head (grouped per-query-head reduction).
        #    seq_of_token(t) := the unique s with subsequence_begins[s] <= t < subsequence_begins[s+1].
        scores = {}
        for t in range(tokens):
            s = seq_of_token(t, subsequence_begins)
            pos = past_lens[s] + (t - subsequence_begins[s])
            n_complete = (pos + 1) // r
            for sh in range(Hsel):
                for b in range(n_complete):
                    phys = block_indices[block_indices_begins[s] + (b * r) // page_size]
                    slot_in_block = (b * r % page_size) // r
                    score = 0.0
                    for kv in range(Hik):
                        if (kv * Hsel) // Hik != sh:
                            continue
                        bar = summary_cache[phys, kv, slot_in_block, :]            # [Di]
                        q_h = q[t, kv * G:(kv + 1) * G, :]                         # [G, Di]
                        # ReLU per index-query head on the dot product, then sum over heads (reference: relu(q·k).sum(heads))
                        score += sum(relu(sum(q_h * bar[None, :], axis=-1)), axis=0)
                    scores[(t, sh, b)] = score * scale

        # 3. Top-k over causally-valid complete blocks per selection head; stable tie-break, ascending order.
        sel_indices = full((tokens, Hsel, Kmax), -1, dtype=int32)
        sel_count   = zeros((tokens, Hsel), dtype=int32)
        for t in range(tokens):
            s = seq_of_token(t, subsequence_begins)
            pos = past_lens[s] + (t - subsequence_begins[s])
            n_complete = (pos + 1) // r
            for sh in range(Hsel):
                k = min(block_topk, n_complete)
                scores_arr = np.array([scores[(t, sh, b)] for b in range(n_complete)])  # [n_complete]
                order = np.argsort(-scores_arr, kind='stable')
                picked = sort(order[:k])                 # strictly ascending block indices
                sel_indices[t, sh, :k] = picked
                sel_count[t, sh] = k
                if (pos + 1) % r != 0:
                    inc = (pos + 1) // r
                    sel_indices[t, sh, k] = inc
                    sel_count[t, sh] = k + 1
        return sel_indices, sel_count


**Attributes**

* *compress_ratio*

  * **Description**: :math:`r`, the number of raw token keys pooled into one block key.
  * **Type**: ``int``
  * **Default value**: ``4``
  * **Required**: *no*
  * **Constraints**: ``compress_ratio >= 1``. ``page_size % compress_ratio == 0``.

* *block_topk*

  * **Description**: Maximum number of *complete* blocks selected per selection head. The output width is
    ``block_topk + 1`` (the extra slot is the causal-diagonal / incomplete block).
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``block_topk >= 1``.

* *h_sel*

  * **Description**: Number of selection heads :math:`H_{sel}` in the emitted contract. Must be ``1`` or
    ``H_ik`` (the index-key head count taken from ``k_idx``). Each index-query head :math:`h` is mapped to a
    selection head via :math:`\mathrm{sel}(h) = \lfloor \mathrm{kv}(h) \cdot H_{sel} / H_{ik} \rfloor`, where
    :math:`\mathrm{kv}(h) = \lfloor h / G \rfloor` and :math:`G = H_{iq} / H_{ik}`.
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*
  * **Constraints**: ``h_sel >= 1``; ``h_sel == 1`` (QSA default) or ``h_sel == H_ik``.

* *indexer_rotary_dim*

  * **Description**: RoPE dimension used by the indexer (partial rotary). Only the first
    ``indexer_rotary_dim`` dims of a block summary are rotated; the trailing
    ``indexer_head_dim - indexer_rotary_dim`` dims are passed through unrotated.
  * **Type**: ``int``
  * **Required**: *yes*

* *indexer_head_dim*

  * **Description**: Head dimension :math:`D_{idx}` of the indexer.
  * **Type**: ``int``
  * **Required**: *yes*

* *page_size*

  * **Description**: Number of tokens per physical page of the shared main PagedAttention cache. Must be a
    multiple of ``compress_ratio`` (this is what makes Prefix Caching page-aligned and lossless).
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``page_size % compress_ratio == 0``.

* *ring_capacity*

  * **Description**: Number of tokens per pending ring slot, :math:`C`. It is a real functional parameter:
    ``C % compress_ratio == 0`` and ``C >= compress_ratio``. The ring is addressed by logical position
    (``off = pos % C``), so ``C == compress_ratio`` supports **standard greedy decode** (one block per slot),
    while ``C >= r + N`` retains the last :math:`N` tokens of the previous block alongside the current one to
    support rollback of a speculative-decode draft of length :math:`N`.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``ring_capacity % compress_ratio == 0`` and ``ring_capacity >= compress_ratio``.

* *eps*

  * **Description**: Epsilon for RMSNorm.
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*

* *scale*

  * **Description**: Score normalization constant, ``1.0 / sqrt(indexer_head_dim)``.
  * **Type**: ``float``
  * **Required**: *yes*


**Inputs**

* **0**: ``q_idx``
  A 2D tensor of type *T* with shape ``[tokens, H_iq * D_idx]``.
  Projected indexer queries emitted by the external ``FullyConnected`` / ``MatMul`` (``index_qk_proj``), already
  RMSNorm'ed and RoPE'ed (the q-side norm/RoPE live outside the op). **Required.**

* **1**: ``k_idx``
  A 2D tensor of type *T* with shape ``[tokens, H_ik * D_idx]``.
  Current-step projected indexer keys, emitted by the same external ``index_qk_proj``. The index-key head count
  ``H_ik`` is **derived from this shape** (configurable; never hard-coded to ``1``). Stored position-addressed in
  the owning sequence's pending ring. **Required.**

* **2**: ``k_rope_cos_sin``
  A 2D tensor of type *T* with shape ``[tokens, 2 * indexer_rotary_dim]``.
  Key-side RoPE ``(cos, sin)``, used **only to capture the block-start phase** when a new block begins (the
  ``(pos // r) % (C / r)`` slot of ``indexer_k_start_cos_sin``). Prepared by the caller from the full table using
  the appropriate (possibly M-RoPE) key coordinates. **Required.**

* **3**: ``summary_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, H_ik, page_size // compress_ratio, D_idx]``.
  Mean-pooled + RMSNorm'ed + block-start-RoPE'ed summary keys of each completed block, addressed through the
  shared main PA page table (``block_indices`` / ``block_indices_begins``). The ``H_ik`` head count is
  configurable, never hard-coded to ``1``. Updated in place. **Required.**

* **4**: ``indexer_k_cache``
  A 4D tensor of type *T* with shape ``[num_slots, H_ik, ring_capacity, D_idx]``.
  Per-sequence ring buffer holding the projected, unrotated keys of the recent (possibly incomplete) blocks,
  addressed by logical position (``off = pos % ring_capacity``). The ``H_ik`` head count is configurable, never
  hard-coded to ``1``. **Required.**

* **5**: ``indexer_k_start_cos_sin``
  A 3D tensor of type *T* with shape ``[num_slots, ring_capacity // compress_ratio, 2 * indexer_rotary_dim]``.
  RoPE ``(cos, sin)`` at the block-start position of each possible pending block start within the ring, addressed
  by ``(pos // compress_ratio) % (ring_capacity / compress_ratio)``. Refreshed at the first token of each block.
  **Required.**

* **6**: ``k_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``.
  RMSNorm weight for pooled block keys. **Required.**

* **7**: ``slot_mapping``
  A 1D tensor of type *T_IND* with shape ``[num_sequences]``.
  Maps each sequence to its pending ring slot. **Required.**

* **8**: ``block_indices``
  A 1D tensor of type *T_IND* with shape ``[total_logical_blocks]``.
  The main PagedAttention logical-to-physical block map; also addresses ``summary_cache``. **Required.**

* **9**: ``block_indices_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits ``block_indices`` among sequences: sequence ``s`` owns blocks
  ``block_indices[block_indices_begins[s] : block_indices_begins[s+1]]``. **Required.**

* **10**: ``subsequence_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits the token stream among sequences: sequence ``s`` uses tokens
  ``[subsequence_begins[s] : subsequence_begins[s + 1]]``. **Required.**

* **11**: ``past_lens``
  A 1D tensor of type *T_IND* with shape ``[num_sequences]``.
  Logical KV length of each sequence before this step. When Prefix Caching matches a prompt prefix, this is
  advanced by the matched prefix length, so the op skips the cached prefix pages. The global logical position of a
  token is ``past_lens[s] + (t - subsequence_begins[s])``. **Required.**


**Outputs**

* **0**: ``sel_indices``
  A 3D tensor of type *T_IND* with shape ``[tokens, H_sel, block_topk + 1]``.
  Selected *block* indices per selection head as a **strictly ascending, duplicate-free** sequence in the
  contiguous prefix ``sel_indices[:sel_count]`` (the causal-diagonal / incomplete block, when present, is appended
  immediately after the selected complete blocks and is always greater than them, so the prefix stays ascending).
  ``-1`` padded only past ``sel_count`` (no interior holes). **Required.**

* **1**: ``sel_count``
  A 2D tensor of type *T_IND* with shape ``[tokens, H_sel]``.
  Number of valid selected blocks per selection head per query. **Required.**


**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` or ``int64``.
* ``tokens``, ``num_sequences`` and ``subsequence_begins`` must be consistent:
  ``subsequence_begins[0] == 0`` and ``subsequence_begins[-1] == tokens``.
* ``block_indices_begins[0] == 0`` and ``block_indices_begins[-1] == len(block_indices)`` (block splits).
* ``q_idx`` last dimension equals ``H_iq * indexer_head_dim``; ``k_idx`` last dimension equals
  ``H_ik * indexer_head_dim``, with both head counts derived from the shapes and ``H_iq % H_ik == 0`` and
  ``H_sel`` in ``{1, H_ik}``.
* ``page_size % compress_ratio == 0``; ``summary_cache`` dim 2 equals ``page_size // compress_ratio``, dim 1
  equals ``H_ik``, and dim 0 equals ``num_blocks`` (shared with the main KV ``key_cache`` dim 0).
* ``slot_mapping`` length equals ``num_sequences``; ``indexer_k_cache`` dim 0 equals ``num_slots``, dim 1 equals
  ``H_ik``, and dim 2 equals ``ring_capacity`` (with ``ring_capacity % compress_ratio == 0`` and
  ``ring_capacity >= compress_ratio``).
* ``indexer_k_start_cos_sin`` dim 0 equals ``num_slots``, dim 1 equals ``ring_capacity // compress_ratio``, and dim 2
  equals ``2 * indexer_rotary_dim``.
* ``sel_indices`` last dimension ``K_max`` equals ``block_topk + 1``; ``sel_count[t, sh] <= K_max``, and the valid
  prefix ``sel_indices[t, sh, :sel_count[t, sh]]`` is **strictly ascending** and contains no ``-1`` entries.
* When the selection for a row is empty (``sel_count == 0``), the consumer must produce a zero output vector for
  that row.


**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Example**

The example below shows a single query token at logical position ``9`` (``past_lens[0] = 9``, one token in the
step) with ``compress_ratio = 4``, ``block_topk = 2`` (so ``K_max = block_topk + 1 = 3``), ``h_sel = 1``,
``H_iq = 4``, ``H_ik = 1``, ``page_size = 16``, ``ring_capacity = 4``. The history written before this step covers
blocks ``{0:[0..3], 1:[4..7]}`` (complete) and block ``2:[8]`` (the causal-diagonal / incomplete block).
Causally-valid complete blocks for ``pos = 9`` are those with ``b < (9+1)//4 = 2``, i.e. ``{0, 1}``. After a stable
top-2 by score over ``{0, 1}`` the selection head emits the top-2 complete blocks in **strictly ascending** order
``{0, 1}`` plus the causal-diagonal block ``2`` written immediately after them, yielding
``sel_indices = [[[0, 1, 2]]]`` and ``sel_count = [[3]]``. With ``H_iq = 4`` and ``H_ik = 1``, all four
index-query heads participate in the single selection head's score.

.. code-block:: xml
   :force:

   <layer ... type="PagedQSAIndexer">
       <data compress_ratio="4" block_topk="2" h_sel="1" indexer_head_dim="128"
             indexer_rotary_dim="128" page_size="16" ring_capacity="4" eps="1e-6"
             scale="0.088388"/>
       <input>
           <port id="0">   <!-- q_idx: [tokens, H_iq*D_idx] -->
               <dim>1</dim><dim>512</dim>
           </port>
           <port id="1">   <!-- k_idx: [tokens, H_ik*D_idx] -->
               <dim>1</dim><dim>128</dim>
           </port>
           <port id="2">   <!-- k_rope_cos_sin: [tokens, 2*rotary_dim] -->
               <dim>1</dim><dim>256</dim>
           </port>
           <port id="3">   <!-- summary_cache: [num_blocks, H_ik, page_size//r, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="4">   <!-- indexer_k_cache: [num_slots, H_ik, ring_capacity, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="5">   <!-- indexer_k_start_cos_sin: [num_slots, C//r, 2*rotary_dim] -->
               <dim>1</dim><dim>1</dim><dim>256</dim>
           </port>
           <port id="6">   <!-- k_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
           <port id="7">   <!-- slot_mapping: [num_sequences] -->
               <dim>1</dim>
           </port>
           <port id="8">   <!-- block_indices: [total_logical_blocks] -->
               <dim>1</dim>
           </port>
           <port id="9">   <!-- block_indices_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="10">  <!-- subsequence_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
           <port id="11">  <!-- past_lens: [num_sequences] -->
               <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="12" precision="I32">  <!-- sel_indices: [tokens, H_sel, block_topk+1] -->
               <dim>1</dim><dim>1</dim><dim>3</dim>
           </port>
           <port id="13" precision="I32">  <!-- sel_count: [tokens, H_sel] -->
               <dim>1</dim><dim>1</dim>
           </port>
       </output>
   </layer>