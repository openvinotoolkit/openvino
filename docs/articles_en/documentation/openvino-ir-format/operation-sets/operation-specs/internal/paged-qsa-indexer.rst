.. {#openvino_docs_ops_internal_PagedQSAIndexer}

PagedQSAIndexer
===============

.. meta::
  :description: Learn about PagedQSAIndexer - a paged, model-specific plan producer for Qwen Sparse Attention (QSA) under continuous batching / PagedAttention page tables. It maintains full-history raw-K and incremental summary caches over the main page table and emits complete-block selections with an optional causal-tail marker.

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
compression chain* (per-token or pooled key mean-pool, RMSNorm, block-start RoPE), maintains the full-history
``indexer_raw_k_cache`` and incremental ``summary_cache``, and emits a clean ``(sel_indices, sel_count)``
block-selection contract. It **supports page-aligned prefix caching** (see "Prefix Caching" below): because
``page_size % compress_ratio == 0``, cached physical pages already have their ``summary_cache`` populated, so a
matched prompt prefix skips summary recomputation for the cached pages.

**Data flow and cache organization**

The K-side history is maintained as two in-place paged caches. The full raw-K cache and summary cache use the
main PagedAttention page table and are passed as explicit inputs:

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

- ``indexer_raw_k_cache`` ``[num_blocks, H_ik, page_size, D_idx]``: all projected, unnormalized, unrotated
  indexer keys stored at their logical token positions. For sequence ``s`` and logical token ``pos``, the operation writes
  ``indexer_raw_k_cache[phys, :, pos % page_size, :]`` where
  ``phys = block_indices[block_indices_begins[s] + pos // page_size]``. The second dimension is the configurable
  index-key head count ``H_ik``. This is full-history storage, not a per-sequence ring: it removes ring wraparound,
  capacity sizing, and cross-block slot-alias hazards, at the cost of memory proportional to the full sequence.
- ``rotary_cos_sin`` ``[max_positions, 2 * indexer_rotary_dim]``: shared RoPE table. Query rows are selected by
  ``position_ids``; summary rows are selected by logical block start ``block_id * compress_ratio``.

A block is *completed* once exactly :math:`r` raw keys have been appended. The flush reads the block's contiguous
raw-key slice from the shared paged cache and writes its pooled, normalized, logically block-start-rotated summary
into ``summary_cache``. The last, partially-filled block is not summarized or selected as a complete block; its
logical ID is appended as the optional final tail marker, and the paired sparse-attention consumer handles only its
visible causal prefix.

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
    restored logical length. No pending RoPE-start state is required because summary phase is derived from the logical
    block ID and shared rotary table.

  **Logical positions and RoPE coordinates**

The op separates the *logical KV index* used for block addressing and causal filtering from the *RoPE coordinate*
used for positional rotation. The **logical position** of a token is derived from ``past_lens`` and the sequence
partition (``subsequence_begins``): for sequence ``s`` and token ``t``,
:math:`pos = past\_lens[s] + (t - subsequence\_begins[s])`. This logical position drives (a) which blocks are
causally valid and (b) the raw-K page/offset. Query RoPE uses ``position_ids[t]`` and summary RoPE uses
the logical block-start row ``b * r`` from ``rotary_cos_sin``. A block completed in a later invocation therefore
needs no saved phase tensor.

**Position-coordinate extensions.** A model whose summary RoPE coordinate is not the logical block start (for
example, a multi-axis or otherwise non-monotonic position scheme) needs a versioned extension that supplies the
block-start coordinate or phase and persists pending-block metadata across calls. The default contract does not
infer such coordinates from the page table or silently add phase state.

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
  ``K_max = block_topk + 1``. The extra slot is reserved for the query's current incomplete causal-tail block.
- **Tie-breaking and ordering**: among equal scores the *smaller block index* (earlier block) wins; the op
  computes the selection as ``argsort(-scores, kind='stable')``, takes the first ``k`` entries, and emits them in
  **strictly ascending** block order (``sort(order[:k])``), yielding a deterministic, implementation-independent,
  monotone result. The valid output prefix ``sel_indices[t, sh, :sel_count[t, sh]]`` is sorted ascending.
- **Causal-tail block ID**: when ``(p+1) % r != 0``, append ``floor((p+1)/r)`` after the selected complete-block
  IDs and increment ``sel_count``. This final ID identifies the block whose visible token prefix is processed by
  the consumer as the causal tail; it is not attended as a complete selected block.

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
        # rotary_cos_sin:   [max_positions, 2 * indexer_rotary_dim]
        # summary_cache:    [num_blocks, H_ik, page_size // r, D_idx]
        # indexer_raw_k_cache: [num_blocks, H_ik, page_size, D_idx] (full history, shared page table)
        # q_norm_weight, k_norm_weight: [D_idx]
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
        Kmax = block_topk + 1
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
                    bar = apply_partial_rope(nb, rotary_cos_sin[b * r], indexer_rotary_dim)
                    slot_in_block = (b * r % page_size) // r
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
                if (pos + 1) % r != 0:
                    sel_indices[t, sh, k] = (pos + 1) // r
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
    ``block_topk + 1``; when the query is inside an incomplete block, the final slot contains that block's logical
    ID as a causal-tail marker. The sparse-attention consumer processes its visible token prefix, not the whole
    block as another selected complete block.
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
  A 2D tensor of type *T* with shape ``[max_positions, 2 * indexer_rotary_dim]``. Shared table used for query
  positions and logical summary block starts. **Required.**

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


**Outputs**

* **0**: ``sel_indices``
  A 3D tensor of type *T_IND* with shape ``[tokens, H_sel, block_topk + 1]``.
  Selected complete-block IDs followed, when the query is inside an incomplete block, by that causal-tail block ID.
  For query ``t`` and selection head ``sh``, the valid prefix
  ``sel_indices[t, sh, :sel_count[t, sh]]`` is **strictly ascending and duplicate-free**; ``-1`` is padded only
  after the prefix (no interior holes). The final tail ID is metadata for causal-tail processing, not another
  complete block to attend. **Required.**

* **1**: ``sel_count``
  A 2D tensor of type *T_IND* with shape ``[tokens, H_sel]``.
  Number of valid output entries (complete-block IDs plus an optional causal-tail marker) per selection head per
  query. **Required.**


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
* ``indexer_raw_k_cache`` dim 0 equals ``num_blocks`` (shared with main KV cache), dim 1 equals ``H_ik``, and dim 2
  equals ``page_size``; it stores raw K for the full logical history, with no ring-capacity attribute.
* ``position_ids`` length equals ``tokens``; ``rotary_cos_sin`` dim 1 equals ``2 * indexer_rotary_dim`` and covers
  the query positions and completed-block start positions used in the invocation.
* ``sel_indices`` last dimension ``K_max`` equals ``block_topk + 1``; ``sel_count[t, sh] <= K_max`` and is at most
  ``min(block_topk, n_complete) + tail_present`` per query, where ``tail_present`` is one iff
  ``(abs_pos + 1) % compress_ratio != 0``. The valid
  prefix ``sel_indices[t, sh, :sel_count[t, sh]]`` is **strictly ascending** and contains no ``-1`` entries.
* When the selection for a row is empty (``sel_count == 0``), the consumer must produce a zero output vector for
  that row.

**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Example**

The example below shows a single query token at logical position ``9`` (``past_lens[0] = 9``, one token in the
step) with ``compress_ratio = 4``, ``block_topk = 2`` (so ``K_max = block_topk + 1 = 3``), ``h_sel = 1``,
``H_iq = 4``, ``H_ik = 1``, and ``page_size = 16``. The history written before this step covers
blocks ``{0:[0..3], 1:[4..7]}`` (complete) and block ``2:[8]`` (incomplete). Causally-valid complete blocks for
``pos = 9`` are ``{0, 1}``. After a stable top-2 by score, the selection head emits ``sel_indices = [[[0, 1, 2]]]``
and ``sel_count = [[3]]`` in strictly ascending order. The final ID ``2`` identifies the partial causal tail and
is not attended as a complete block. With ``H_iq = 4`` and ``H_ik = 1``, all four
index-query heads participate in the single selection head's score.

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
             <port id="3">   <!-- rotary_cos_sin: [max_positions, 2*rotary_dim] -->
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
       </input>
       <output>
             <port id="12" precision="I32">  <!-- sel_indices: [tokens, H_sel, block_topk + 1] -->
               <dim>1</dim><dim>1</dim><dim>3</dim>
           </port>
           <port id="13" precision="I32">  <!-- sel_count: [tokens, H_sel] -->
               <dim>1</dim><dim>1</dim>
           </port>
       </output>
   </layer>