.. {#openvino_docs_ops_internal_PagedQSAIndexer}

PagedQSAIndexer
===============

.. meta::
  :description: Learn about PagedQSAIndexer - a paged, model-specific plan producer for Qwen Sparse Attention (QSA) under continuous batching / PagedAttention page tables. It maintains full-history raw-K and incremental summary caches over the main page table and emits selections containing only complete blocks.

**Versioned name**: *PagedQSAIndexer*

**Category**: *Internal*

**Short description**:
The *PagedQSAIndexer* is the **model-specific plan producer** for Qwen Sparse Attention (QSA) in its
**paged** form, used under **continuous batching / PagedAttention page tables**. It consumes externally projected
indexer queries and keys, applies query RMSNorm/RoPE, appends raw keys, generates summaries for completed blocks,
and produces block selections. Unlike the standard
:doc:`QSAIndexer <qsa-indexer>` (a pure Functional State Interface with external ``ReadValue``/``Assign`` state),
*PagedQSAIndexer* owns its recurrent caches **in place** as explicitly-passed paged tensors and reads the shared
main PagedAttention page table (``block_indices`` / ``block_indices_begins``). It performs the same *K-side
compression chain* (per-token or pooled key mean-pool, RMSNorm, RoPE using the real recorded ``position_ids`` of
each completed block's first logical token), maintains the full-history ``indexer_raw_k_cache``,
``indexer_position_cache`` and incremental ``summary_cache``, and emits a clean ``(sel_indices, sel_count)``
block-selection contract. It **supports page-aligned prefix caching** (see "Prefix Caching" below): because
``page_size % compress_ratio == 0``, cached physical pages already have their ``summary_cache`` populated, so a
matched prompt prefix skips summary recomputation for the cached pages.

**Data flow and cache organization**

The K-side history is maintained as two in-place paged caches. The full raw-K cache and summary cache use the
main PagedAttention page table and are passed as explicit inputs:

- ``summary_cache`` ``[num_blocks, H_ik, page_size / compress_ratio, D_idx]``: the mean-pooled, RMSNorm'ed,
  real-position-RoPE'ed summary key of each *completed* block. This is the incrementally-updated artifact that
  makes the per-token update :math:`O(1)`. It is addressed by the **same page table** as the main PagedAttention
  KV cache (``block_indices`` / ``block_indices_begins``). For sequence ``s``, block ``b`` (covering logical
  tokens ``[b*r, (b+1)*r - 1]``) is stored at the physical location
  ``summary_cache[phys, :, slot_in_block, :]`` with
  ``phys = block_indices[block_indices_begins[s] + (b * r) // page_size]`` and
  ``slot_in_block = (b * r % page_size) // r``. The second dimension is the **configurable index-key head count
  ``H_ik``** (never hard-coded to ``1``). This requires
  ``page_size % compress_ratio == 0`` so every summary slot maps to a fixed position inside one physical page.

- ``indexer_raw_k_cache`` ``[num_blocks, H_ik, page_size, D_idx]``: all projected, unnormalized, unrotated
  indexer keys stored at their logical token positions. For sequence ``s`` and logical token ``pos``, the operation writes
  ``indexer_raw_k_cache[phys, :, pos % page_size, :]`` where
  ``phys = block_indices[block_indices_begins[s] + pos // page_size]``. The second dimension is the configurable
  index-key head count ``H_ik``. This is full-history storage, not a per-sequence ring: it removes ring wraparound,
  capacity sizing, and cross-block slot-alias hazards, at the cost of memory proportional to the full sequence.
- ``indexer_position_cache`` ``[num_blocks, page_size / compress_ratio]``: one recorded ``position_ids`` value per
  QSA block -- the position of that block's first logical token -- addressed through the same page table and at
  the same per-page granularity as ``summary_cache``. Written once, when that first token is processed (which may
  be an earlier invocation than the one that completes the block), and read back once when the block completes.
  This replaces inferring the coordinate from a synthetic ``block_id * compress_ratio`` count, so a block whose
  first token does not fall on a monotonic, gapless position (e.g. a multi-axis or otherwise non-monotonic mRoPE
  scheme) still rotates correctly. Only one value per block is kept, not one per raw key: nothing else is ever
  read back from this cache.
- ``rotary_cos_sin`` ``[max_positions, indexer_rotary_dim]``: shared NEOX RoPE table, row ``p`` holding interleaved
  ``(cos_j, sin_j)`` pairs for ``j < indexer_rotary_dim / 2``. Query rows are selected by
  ``position_ids``; summary rows are selected by the real recorded position read from
  ``indexer_position_cache``.

A block is *completed* once exactly :math:`r` raw keys have been appended. The flush reads the block's contiguous
raw-key slice from the shared paged cache and writes its pooled, normalized, real-position-rotated summary
into ``summary_cache``. The last, partially-filled block is not summarized and is never included in this
operation's selection. SparseAttention independently derives and processes the causal tail for each query from its
logical position.

**Prefix Caching**

Because ``page_size % compress_ratio == 0``, a cached physical page always has every ``summary_cache`` slot for
its completed blocks already populated — the summary layout is page-aligned. Prefix caching is **page-aligned**:
the scheduler only matches prefixes whose length is a whole multiple of ``page_size``.

.. note::
   **Why page-aligned matching is required.** Since ``page_size % compress_ratio == 0``, a matched prefix of any
    ``matched_len`` satisfies ``matched_len % compress_ratio == 0``, so the new sequence starts at a complete-block
    boundary. Its shared physical pages contain both raw-K history and summaries, and are only **read**; the suffix
    writes into **new** pages, so no copy-on-write is needed. Prefix matching remains page-aligned because the page
    table shares whole physical pages, not partial pages. If the matched prefix is not a whole multiple of
    ``page_size``, it must be **rounded down** to the page boundary (``matched_len := matched_len - matched_len %
    page_size``); otherwise the suffix would need to write into a shared physical page.

**Execution flow.** When a new request's prompt shares a page-aligned prefix with a previously-cached one, the
scheduler:

1. Detects the shared prefix length, **rounds it down to a multiple of ``page_size``**, and advances ``past_lens``
   by that length (``past_lens[B] = matched_len``). The shared KV pages — and therefore their ``summary_cache``
   rows — are reused by the page table; ``block_indices[B]`` points at the shared pages first, then at new pages
   for the suffix.
2. The op processes only the *unmatched suffix* (logical positions ``>= past_lens[B]``), writing raw keys and
    newly completed summaries into the **new** suffix pages. No COW, no overwrite of the shared prefix pages.
3. The op **skips recomputing summaries for the cached prefix pages**: it only scores against the existing summary
    keys of the shared prefix blocks (reading them directly) plus the newly-written suffix summaries. The matching
    raw-K pages are retained as full history and are likewise read-only for the shared prefix.

This is *lossless* with respect to the sparse plan **given deterministic summary computation**: the summary keys
for the prefix blocks are byte-identical to what a fresh prefill would have produced, so the emitted
``(sel_indices, sel_count)`` for the suffix is exactly what it would be had the prefix been re-computed. The cost
of prefix caching is therefore only the *scoring* over the (already-summarized) prefix blocks, not the
*construction* of those summaries.

**Worked example.** Let ``page_size = 16`` and ``compress_ratio = 4``. Request A writes 40 tokens:
page ``P0 = [0..15]`` (blocks 0–3), page ``P1 = [16..31]`` (blocks 4–7), page ``P2 = [32..39]`` (blocks 8–9, the
last page partially filled). Request B shares the first 35 tokens of A. Whole pages shared: ``32 = floor(35 / 16) *
16``, so ``past_lens[B] = 32`` and ``block_indices[B] = [P0, P1, Pnew]``. B's suffix is tokens ``32..`` in ``Pnew``;
it processes logical positions ``32, 33, ...`` (fresh raw-K writes to the suffix pages). At logical position ``pos = 40``
(block ``10``), B scores blocks ``0..7`` from shared pages ``P0/P1`` and blocks ``8, 9`` from its own ``Pnew``
(just flushed).

**Why page-aligned rounding is mandatory.** If one attempted to share ``34`` tokens, the page table could not
represent that prefix as whole shared pages: the page containing tokens ``32, 33`` would also contain positions
that B must write independently. Sharing that page would alias the two sequences' writable raw-K/summary storage;
copying or allocating a private page is required. Therefore the prefix length is rounded down to the page boundary.

.. note::
    **Decode mode and rollback.** There is no ``ring_capacity`` and no raw-K ring window: rollback is not limited
    by a configured ring size, provided the scheduler retains the raw-K pages and restores the valid summary length.
    Replaying from a smaller ``past_lens`` overwrites raw K at the same logical positions and recomputes summaries
    for blocks completed by the replay. Summaries beyond the rollback point must be invalidated or hidden by the
    restored logical length. Replaying also overwrites ``indexer_position_cache`` at the same logical positions,
    so a recomputed summary always rotates with the position recorded by the replay, not a stale one.

  **Logical positions and RoPE coordinates**

The op separates the *logical KV index* used for block addressing and causal filtering from the *RoPE coordinate*
used for positional rotation. The **logical position** of a token is derived from ``past_lens`` and the sequence
partition (``subsequence_begins``): for sequence ``s`` and token ``t``,
:math:`pos = past\_lens[s] + (t - subsequence\_begins[s])`. This logical position drives (a) which blocks are
causally valid and (b) the raw-K page/offset. Query RoPE uses ``position_ids[t]``; summary RoPE uses the **real
recorded position** of the block's first logical token, read from ``indexer_position_cache`` at that block's own
slot rather than inferred as ``b * r``. A block completed in a later invocation still needs no separate pending
phase tensor beyond this: ``indexer_position_cache`` is written once, in whichever invocation first appends the
block's first token, so by the time the block completes (possibly several invocations later) that position is
already resident. An invocation that both starts and completes a block in the same call avoids a same-work-group
global write-then-read round trip (observed to race without an explicit global coherence fence) by also staging
the value through local (SLM) memory for that call's own use.

**Non-monotonic / multi-axis position schemes.** Because the block-start RoPE coordinate is read from
``indexer_position_cache`` instead of inferred as ``block_id * compress_ratio``, a model whose ``position_ids``
is not monotonic or gapless (for example, an mRoPE scheme where multiple tokens share one axis's rank) rotates
summaries correctly without any extra block-grouping change: block membership (which ``r`` raw keys form one
block) still follows the plain logical-token count, only the RoPE coordinate applied to the pooled result is the
real recorded value.

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
heads*, matching the QSA scoring semantics. Under QSA (:math:`H_{iq} = 4`, :math:`H_{ik} = 1`,
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
  ``K_max = block_topk``. ``sel_count`` counts complete-block IDs only and is at most
  ``min(block_topk, n_complete)``; no tail ID or tail token indices are emitted.
- **Tie-breaking and ordering**: among equal scores the *smaller block index* (earlier block) wins; the op
  computes the selection as ``argsort(-scores, kind='stable')``, takes the first ``k`` entries, and emits them in
  **strictly ascending** block order (``sort(order[:k])``), yielding a deterministic, implementation-independent,
  monotone result. The valid output prefix ``sel_indices[t, sh, :sel_count[t, sh]]`` is sorted ascending.
- **Causal tail**: it is not part of the selection output. SparseAttention derives
  ``tail_start = floor((p+1)/r) * r`` for each query and attends token positions ``tail_start..p`` inclusive;
  this span is empty at a complete-block boundary.

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def paged_qsa_indexer(q_idx, k_idx, position_ids, rotary_cos_sin,
                summary_cache, indexer_raw_k_cache, q_norm_weight, k_norm_weight,
                block_indices, block_indices_begins,
                          subsequence_begins, past_lens,
                          *, compress_ratio, block_topk, h_sel,
                          indexer_rotary_dim, indexer_head_dim, page_size,
                          num_sequences, eps, scale):
        # q_idx:            [tokens, H_iq * D_idx]   (raw external projection; norm/RoPE applied here)
        # k_idx:            [tokens, H_ik * D_idx]   (raw external projection)
        # position_ids:     [tokens]                (query RoPE table indices)
        # rotary_cos_sin:   [max_positions, indexer_rotary_dim], interleaved (cos_j, sin_j)
        # summary_cache:    [num_blocks, H_ik, page_size // r, D_idx]
        # indexer_raw_k_cache: [num_blocks, H_ik, page_size, D_idx] (full history, shared page table)
        # q_norm_weight, k_norm_weight: [D_idx]
        # block_indices:    [total_logical_blocks]   logical -> physical block map (shared with main PA)
        # block_indices_begins: [num_sequences + 1]  split pointers into block_indices (len == total_logical_blocks)
        # subsequence_begins:   [num_sequences + 1]  split pointers into the token stream (len == tokens)
        # past_lens:        [num_sequences]  (logical length of each sequence before this step)
        # returns: sel_indices [tokens, H_sel, K_max]  (int32; -1 = pad), K_max = block_topk
        #          sel_count   [tokens, H_sel]         (int32)
        tokens = q_idx.shape[0]
        Hiq = q_idx.shape[1] // indexer_head_dim      # index-query heads, derived from the shape
        Hik = k_idx.shape[1] // indexer_head_dim      # index-key heads, derived from the shape
        Hsel = h_sel
        r = compress_ratio
        Kmax = block_topk
        G = Hiq // Hik                                # index-query heads per index-key head
        assert page_size % r == 0 and Hiq % Hik == 0 and Hsel in (1, Hik)

        k = reshape(k_idx, (tokens, Hik, indexer_head_dim))
        q = reshape(q_idx, (tokens, Hiq, indexer_head_dim))
        q = rms_norm(q, q_norm_weight, eps)
        q = apply_partial_rope(q, rotary_cos_sin[position_ids], indexer_rotary_dim)
        # 1. Per-token K-side update, addressed by logical page and offset. Full-history raw K
        #    means no ring wraparound or ring-capacity constraint.
        for s in range(num_sequences):
              base = past_lens[s]                       # logical position of the first token of this chunk
              for t in range(subsequence_begins[s], subsequence_begins[s + 1]):
                pos = base + (t - subsequence_begins[s])    # global logical position
                phys = block_indices[block_indices_begins[s] + pos // page_size]
                off = pos % page_size
                indexer_raw_k_cache[phys, :, off, :] = k[t]  # [Hik, Di] (H_ik configurable)
                if pos % r == r - 1:
                    # Block `b = pos // r` is now complete: read its page-local raw-K slice,
                    # mean-pool (fp32), RMSNorm, partial RoPE, then flush the summary.
                    b = pos // r
                    block_phys = block_indices[block_indices_begins[s] + (b * r) // page_size]
                    raw_start = (b * r) % page_size
                    pooled = mean(indexer_raw_k_cache[block_phys, :, raw_start:raw_start + r, :]
                                  .astype(float32), axis=1).astype(k.dtype)       # [Hik, Di]
                    nb = rms_norm(pooled, k_norm_weight, eps)                     # [Hik, Di]
                    slot_in_block = (b * r % page_size) // r
                    bar = apply_partial_rope(nb, rotary_cos_sin[indexer_position_cache[block_phys, slot_in_block]],
                                             indexer_rotary_dim)
                    summary_cache[block_phys, :, slot_in_block, :] = bar        # [Hik, Di]

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
        return sel_indices, sel_count


**Attributes**

* *compress_ratio*

  * **Description**: :math:`r`, the number of raw token keys pooled into one block key.
  * **Type**: ``int``
  * **Default value**: ``4``
  * **Required**: *no*
  * **Constraints**: ``compress_ratio >= 1``. ``page_size % compress_ratio == 0``.

* *block_topk*

  * **Description**: Maximum number of *complete* blocks selected per selection head and the output width
    ``K_max``. The causal tail is not part of this output and does not consume a selection slot.
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
  Raw projected indexer queries before indexer RMSNorm and RoPE. The operation applies both internally. **Required.**

* **1**: ``k_idx``
  A 2D tensor of type *T* with shape ``[tokens, H_ik * D_idx]``.
  Current-step projected indexer keys. The index-key head count ``H_ik`` is derived from this shape and may be
  greater than one. Keys are written to the full-history
  raw-K cache at the logical token position. **Required.**

* **2**: ``position_ids``
  A 1D integer tensor with shape ``[tokens]`` containing the rotary-table index for each query token. **Required.**

* **3**: ``rotary_cos_sin``
  A 2D tensor of type *T* with shape ``[max_positions, indexer_rotary_dim]``. Row ``p`` stores the
  ``indexer_rotary_dim / 2`` NEOX (rotate-half) frequencies of position ``p`` as interleaved pairs
  ``(cos_0, sin_0, cos_1, sin_1, ...)``. Query rows use ``position_ids``; summary rows use the real recorded
  position read from ``indexer_position_cache``. **Required.**

* **4**: ``summary_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, H_ik, page_size // compress_ratio, D_idx]``.
  Mean-pooled + RMSNorm'ed + block-start-RoPE'ed summary keys of each completed block, addressed through the
  shared main PA page table (``block_indices`` / ``block_indices_begins``). The ``H_ik`` head count is
  configurable, never hard-coded to ``1``. Updated in place. **Required.**

* **5**: ``indexer_raw_k_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, H_ik, page_size, D_idx]``.
  Full-history cache of projected, unnormalized, unrotated keys, addressed through the same ``block_indices`` /
  ``block_indices_begins`` page table as the main KV cache. For logical position ``pos``, the operation writes to physical
  page ``block_indices[block_indices_begins[s] + pos // page_size]`` and offset ``pos % page_size``. The ``H_ik``
  head count is configurable, never hard-coded to ``1``. Updated in place. **Required.**

* **6**: ``q_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``. RMSNorm weight for raw projected indexer queries. **Required.**

* **7**: ``k_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``.
  RMSNorm weight for pooled block keys. **Required.**

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

* **12**: ``indexer_position_cache``
  A 2D tensor of type *T_IND* with shape ``[num_blocks, page_size // compress_ratio]``.
  One recorded ``position_ids`` value per QSA block -- the real position of that block's first logical token --
  addressed through the same page table as ``summary_cache``. Written once, in whichever invocation first appends
  that first token (physical page ``block_indices[block_indices_begins[s] + pos // page_size]``, offset
  ``(pos // compress_ratio) % (page_size // compress_ratio)``), and read back once the block completes, to
  recover the real RoPE coordinate for that block's summary. Updated in place. **Required.**

Inputs 4, 5 and 12 are mutated in place. When they are model inputs, the bound tensors MUST be device-visible
storage of the exact port element type (e.g. a GPU RemoteTensor or USM host tensor); a runtime that would stage
them through a host copy or precision conversion MUST reject the binding instead of silently dropping the update.


**Outputs**

* **0**: ``sel_indices``
  A 3D tensor of type *T_IND* with shape ``[tokens, H_sel, K_max]``, where ``K_max = block_topk``.
  Contains only selected complete-block IDs. For query ``t`` and selection head ``sh``, the valid prefix
  ``sel_indices[t, sh, :sel_count[t, sh]]`` is **strictly ascending and duplicate-free**; ``-1`` is padded only
  after the prefix (no interior holes). SparseAttention derives the causal tail per query; it is not represented
  in this output. **Required.**

* **1**: ``sel_count``
  A 2D tensor of type *T_IND* with shape ``[tokens, H_sel]``.
  Number of selected complete-block IDs per selection head per query, satisfying
  ``0 <= sel_count <= min(block_topk, n_complete)``. Tail tokens are not counted. **Required.**

For the QSA configuration ``H_sel = 1``, the logical selection ABI is ``sel_indices[T, K]`` and
``sel_count[T]``, with ``T = tokens`` and ``K = block_topk``. The explicit ``H_sel`` axis above is retained for
the generalized multi-selection-head form.


**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` (``position_ids`` and the page-table inputs included).
* ``tokens``, ``num_sequences`` and ``subsequence_begins`` must be consistent:
  ``subsequence_begins[0] == 0`` and ``subsequence_begins[-1] == tokens``.
* ``block_indices_begins[0] == 0`` and ``block_indices_begins[-1] == len(block_indices)`` (block splits).
* ``q_idx`` last dimension equals ``H_iq * indexer_head_dim``; ``k_idx`` last dimension equals
  ``H_ik * indexer_head_dim``, with both head counts derived from the shapes and ``H_iq % H_ik == 0`` and
  ``H_sel`` in ``{1, H_ik}``.
* ``page_size % compress_ratio == 0``; ``summary_cache`` dim 2 equals ``page_size // compress_ratio``, dim 1
  equals ``H_ik``, and dim 0 equals ``num_blocks`` (shared with the main KV ``key_cache`` dim 0).
* ``indexer_raw_k_cache`` dim 0 equals ``num_blocks`` (shared with main KV cache), dim 1 equals ``H_ik``, and dim 2
  equals ``page_size``; it stores raw K for the full logical history, with no ring-capacity attribute.
* ``position_ids`` length equals ``tokens``; ``rotary_cos_sin`` dim 1 equals ``indexer_rotary_dim`` and covers
  the query positions and the real recorded positions read from ``indexer_position_cache`` for completed blocks.
* ``indexer_position_cache`` dim 0 equals ``num_blocks`` (shared with the main KV cache and ``indexer_raw_k_cache``)
  and dim 1 equals ``page_size // compress_ratio`` (shared with ``summary_cache``); it must be writable and stores
  one position per completed-or-pending block, not a per-token or ring-capacity tensor.
* ``sel_indices`` last dimension ``K_max`` equals ``block_topk``; ``sel_count[t, sh] <= K_max`` and is at most
  ``min(block_topk, n_complete)`` per query. The valid
  prefix ``sel_indices[t, sh, :sel_count[t, sh]]`` is **strictly ascending** and contains no ``-1`` entries.
* If ``sel_count == 0``, a causal SparseAttention consumer may still produce a nonzero result from its
  automatically-derived tail. It produces zero only when no effective key remains after selection and tail
  handling.

**Types**

* *T*: any floating point type.
* *T_IND*: ``int32``.


**Example**

The example below shows a single query token at logical position ``9`` (``past_lens[0] = 9``, one token in the
step) with ``compress_ratio = 4``, ``block_topk = 2`` (so ``K_max = block_topk = 2``), ``h_sel = 1``,
``H_iq = 4``, ``H_ik = 1``, and ``page_size = 16``. The history written before this step covers
blocks ``{0:[0..3], 1:[4..7]}`` (complete) and block ``2:[8]`` (incomplete). Causally-valid complete blocks for
``pos = 9`` are ``{0, 1}``. After a stable top-2 by score, the selection head emits ``sel_indices = [[[0, 1]]]``
and ``sel_count = [[2]]`` in strictly ascending order. SparseAttention independently derives the tail positions
``[8, 9]`` for this query; the tail is absent from the selection output/count. With ``H_iq = 4`` and ``H_ik = 1``,
all four index-query heads participate in the single selection head's score.

.. code-block:: xml
   :force:

   <layer ... type="PagedQSAIndexer">
       <data compress_ratio="4" block_topk="2" h_sel="1" indexer_head_dim="128"
         indexer_rotary_dim="128" page_size="16" eps="1e-6"
             scale="0.088388"/>
       <input>
           <port id="0">   <!-- q_idx: [tokens, H_iq*D_idx] -->
               <dim>1</dim><dim>512</dim>
           </port>
           <port id="1">   <!-- k_idx: [tokens, H_ik*D_idx] -->
               <dim>1</dim><dim>128</dim>
           </port>
             <port id="2">   <!-- position_ids: [tokens] -->
               <dim>1</dim>
           </port>
             <port id="3">   <!-- rotary_cos_sin: [max_positions, rotary_dim] -->
               <dim>16</dim><dim>256</dim>
             </port>
             <port id="4">   <!-- summary_cache: [num_blocks, H_ik, page_size//r, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
             <port id="5">   <!-- indexer_raw_k_cache: [num_blocks, H_ik, page_size, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>16</dim><dim>128</dim>
           </port>
             <port id="6">   <!-- q_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
             <port id="7">   <!-- k_norm_weight: [D_idx] -->
               <dim>128</dim>
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
           <port id="12">  <!-- indexer_position_cache: [num_blocks, page_size//r], writable -->
               <dim>4</dim><dim>4</dim>
           </port>
       </input>
       <output>
             <port id="13" precision="I32">  <!-- sel_indices: [tokens, H_sel, K_max=block_topk] -->
               <dim>1</dim><dim>1</dim><dim>2</dim>
           </port>
           <port id="14" precision="I32">  <!-- sel_count: [tokens, H_sel] -->
               <dim>1</dim><dim>1</dim>
           </port>
       </output>
   </layer>