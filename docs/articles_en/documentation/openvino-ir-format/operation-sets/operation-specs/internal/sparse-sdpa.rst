.. {#openvino_docs_ops_internal_SparseSDPA}

SparseSDPA
==========

.. meta::
  :description: Learn about SparseSDPA - a unified, model-agnostic sparse attention operation that consumes a clean (sel_indices, sel_count) block-selection contract with 4D SDPA-aligned tensor layouts and performs online-softmax attention over the selected KV blocks without a physical page table.

**Versioned name**: *SparseSDPA*

**Category**: *Internal*

**Short description**:
The *SparseSDPA* operation performs sparse, selected-block multi-head attention over *contiguous,
KVCache-managed* K/V tensors. It is a **model-agnostic plan consumer** over a strictly clean
``(sel_indices, sel_count)`` block-selection contract from a *model-specific* producer (e.g. ``QSAIndexer``),
gathers the KV at the selected blocks, and applies online-softmax attention with GQA head
mapping. It contains **no producer-leaked artifacts**: no ``index_weights``, no output gate, no RoPE/norm
parameters, and no ``r-1`` tail hack. It is intentionally RoPE- and norm-free: all positional and normalization
math is owned by the main attention layer and the upstream indexer. The operation is *stateless with respect to
the selection*: it recomputes attention over the selected set each step.

The operation is specified in the exact tensor language of the `Scaled Dot-Product Attention` (`SDPA-13`)
operation set: the query, key and value inputs adopt the 4D ``[B, H, L, D]`` SDPA layout, and the operation is
*defined* as ``SDPA13`` invoked with ``causal = false`` and an explicit ``combined_mask`` that is the
conjunction of the optional ``attn_mask``, the **bottom-right aligned** generative causal pattern
:math:`kpos \le S - L + l`, and the sparsity plan expanded to a full attention mask. It is **mathematically
equivalent to SDPA-13** over the masked set: an implementation must reproduce the dense SDPA-13 result over that
set, with numerical consistency within a given tolerance (the online-softmax reduction order may differ).

**Detailed description**

*SparseSDPA* decouples *attention execution* from *sparse plan production*. Its inputs are:

- the current query (already projected by the main Q-projection, RMSNorm'ed and RoPE'ed by the attention
  layer),
- dense, history-complete K and V cache tensors produced by a standard ``KVCache`` node, and
- the clean selection produced by an upstream model-specific indexer: per-query-block, per-selection-head block
  indices ``sel_indices`` and their valid counts ``sel_count``, plus the block granularities ``sel_block_size``
  (:math:`B_k`) and ``q_block_size`` (:math:`B_q`).

The operation:

1. **Partitions** the query sequence into query blocks of :math:`B_q` tokens (default :math:`B_q = 1`); every
   query in a query block shares the selection emitted for that block. The number of query blocks is
   :math:`\mathrm{num\_q\_blocks} = \lceil L / B_q \rceil`. When :math:`B_q = 1`, ``num_q_blocks == L`` (one
   selection per query token); when :math:`B_q > 1` (e.g. FlashVSR block sparse with :math:`B_q = 128`), the
   selection is carried once per :math:`B_q`-token query block and replicated across the block's rows.
2. **Gathers** the selected KV blocks for each query block (a block-gather, not a masked matmul over the full
   history — the key bandwidth/FLOP win of sparse attention). The causal-diagonal / incomplete block, when
   emitted by the producer as a valid block index, is gathered as a partial block bounded by the query position
   *inclusive* (:math:`\le pos`).
3. **Scores** each selected key against the query with the attention scale ``1/sqrt(Dh)`` (overridable via the
   ``scale`` input).
4. **Online-softmax** normalizes scores over the selected set, subject to the optional ``attn_mask``.
5. **Weighted-averages** the gathered values and concatenates across heads.

Because the selection is index-based (a fixed-width block buffer per query block), there is no need to materialize
a ``[B, H, L, S]`` boolean mask — the exact source of the bandwidth saving. The op supports group-query attention
(GQA) with ``num_kv_heads <= num_heads`` and a clean mapping from query heads to selection heads.

**The clean selection contract**

The producer emits block-level selection in the same 4D language as the rest of the op, with the **first
selection dimension equal to ``num_q_blocks``**:

- ``sel_indices`` ``[B, H_sel, num_q_blocks, K_max]``: selected *block* indices per query block per selection
  head, as a **strictly ascending, duplicate-free** sequence in the contiguous prefix ``sel_indices[:sel_count]``,
  ``-1`` padded only past ``sel_count``. Here ``num_q_blocks = ceil(L / B_q)``.
- ``sel_count`` ``[B, H_sel, num_q_blocks]``: valid block count per query block per selection head.
- ``sel_block_size`` (attribute, :math:`B_k`): the number of tokens per selected KV block; the consumer gathers
  full blocks of this size (and a partial prefix of the causal-diagonal block, bounded by the query's absolute
  position).
- ``q_block_size`` (attribute, :math:`B_q`, default ``1``): the number of query tokens that share one selection.

The op never re-derives the selection: it gathers exactly the blocks named in the valid prefix of ``sel_indices``
and counted by ``sel_count``, and carries no ``index_weights``, no output gate, and no ``r-1`` tail encoding. The
plan is position-free: causality is a joint producer/consumer responsibility (see "Causal responsibility"), and
the consumer applies the **token-level** causal truncation :math:`kpos \le S - L + l` (including the diagonal)
that is the final guarantee of correctness. An empty selection (``sel_count == 0``, e.g. a query with no
causally-valid block) yields a zero output vector for that row.

**Mask-only plans for unbounded-K_max producers**

Not every producer has a fixed per-row top-:math:`K`. Threshold-based selection — e.g. XAttention's block
scoring, which greedily accumulates the highest-scoring blocks until their *cumulative softmax mass* crosses a
threshold fraction, with no cap on how many blocks that takes — is **data-dependent**: for a near-uniform score
distribution the loop keeps going until almost every block is selected, so the worst-case selected-block count
approaches the full block grid, :math:`K_{max} \to \mathrm{num\_k\_blocks} = \lceil S / B_k \rceil`. Unlike QSA's
fixed ``block_topk + 1``, there is no tighter bound to declare. Forcing such a plan through ``sel_indices`` then
pays for a full-width ``int32`` index per logical block per query row (4 bytes, often carrying no more
information than "every block is selected") instead of the 1 bit a boolean mask would need — the opposite of the
bandwidth win the index-based contract exists to provide.

For this case, the plan **may instead be carried directly through** ``attn_mask`` **in place of**
``sel_indices``/``sel_count``: the producer expands its ``[B, H_sel, num_q_blocks, num_k_blocks]`` boolean
block decision to a tensor broadcastable to ``[B, H_sel, L, S]`` (SDPA-13 boolean-mask semantics, replicated at
``sel_block_size``/``q_block_size`` granularity) and supplies it as ``attn_mask``; ``sel_indices`` and
``sel_count`` are then omitted, and ``combined_mask`` reduces to ``attn_mask`` :math:`\wedge`
:math:`\mathrm{causal}_{S-L+l}` (``expand(plan)`` is trivially all-ones since the plan no longer exists as a
separate index structure). This forgoes the bandwidth benefit a bounded index list gives a consumer that can
skip ungathered blocks entirely, but that benefit was illusory for a producer whose own selection is already
block-dense: expressing it as a mask avoids round-tripping through a same-size-or-larger ``int32`` index buffer
for no gain. This mirrors the GPU implementation of XAttention (``xattn_find_block.cm`` / ``xattn_post_proc.cm``),
which computes exactly a per-(head, q-block, k-block) boolean mask and feeds it to the paged-attention kernel
directly — it is never converted into an index list.

One of the two plan representations must be present: either the ``sel_indices``/``sel_count`` pair, or a
plan-carrying ``attn_mask``. When ``sel_indices``/``sel_count`` are omitted, ``attn_mask`` is **required** (not
optional) and plays both roles at once — ordinary attention masking and sparsity selection — since the two are
joined by the same conjunction regardless of representation.

**Causal responsibility**

Causality is split across the producer and the consumer:

- **Producer (e.g. ``QSAIndexer``) — block-level coarse filter (Block Eligibility).** The
  producer ensures no *future / not-yet-written complete block* is ever placed in the plan: it selects only the
  causally-valid complete blocks (block :math:`b` is eligible iff :math:`(b{+}1) \cdot B_k - 1 \le pos`) plus the
  causal diagonal. The plan itself is position-free, however: the consumer applies the exact per-token truncation.
- **Consumer (this op) — token-level fine filter (Token-level Precision).** The consumer is the final authority
  on token-level causality: it gathers the selected blocks and imposes :math:`kpos \le S - L + l` (including the
  diagonal, self-attended). This is what guarantees exact causal semantics regardless of the producer's block
  granularity.

This division mirrors ``SparsePA``: the producer performs the coarse block filter and the consumer the fine token
filter.

**Equivalence to SDPA-13**

OpenVINO's native `SDPA-13` defines ``causal=true`` with a **top-left aligned** causal mask
(:math:`\mathrm{triu}(-\infty, k{=}1)`), and, when ``causal=true``, it **ignores the supplied ``attn_mask``**.
Generative models / Transformers / PagedAttention, by contrast, use a **bottom-right aligned** causal mask
(:math:`kpos \le S - L + l`). To keep *SparseSDPA* exactly aligned with the generative convention, the op never
delegates causality to ``SDPA-13``'s native ``causal`` flag. Instead it builds a single explicit
``combined_mask`` and invokes the dense op with ``causal = false``:

.. math::

    \mathrm{SparseSDPA}(q, k, v; \mathrm{attn\_mask}, \mathrm{causal}, \mathrm{plan})
        = \mathrm{SDPA13}\!\left(q, k, v,\ \mathrm{attn\_mask} = \mathrm{combined\_mask},\ \mathrm{causal} = \mathrm{false}\right)

where

.. math::

    \mathrm{combined\_mask} = \mathrm{attn\_mask} \wedge \mathrm{causal}_{S-L+l} \wedge \mathrm{expand}(\mathrm{plan})

and

- :math:`\mathrm{causal}_{S-L+l}` is the **bottom-right aligned** lower-triangular mask
  :math:`kpos \le S - L + l` (including the diagonal, i.e. self-attended),
- :math:`\mathrm{expand}(\mathrm{plan})` is the :math:`[B, H_q, L, S]` boolean mask that is ``1`` exactly where
  row ``(b, h, l)`` attends to a KV token inside a block selected by the plan (non-selected blocks set to
  ``0``), and
- :math:`\wedge` is the element-wise conjunction of the three mask sources.

This identity is the *definitional* contract of the op: any implementation must reproduce the dense SDPA-13 result
over the masked set. It is **mathematically equivalent**, with numerical consistency within a given tolerance — the
online-softmax accumulation order may differ from the dense reference, so results are not bit-comparable. Only
when :math:`L == S` (full-sequence prefill) does this coincide with ``SDPA-13``'s own ``causal=true`` behavior; for
:math:`L < S` (decode / incremental), the bottom-right alignment differs from ``SDPA-13``'s top-left
``causal=true``, which is why the mask is always passed explicitly with ``causal = false``.

``attn_mask`` **semantics** follow SDPA-13: it may be supplied either as a **boolean** mask (``False`` positions
are excluded from the softmax) or as a **floating-point additive** mask (``-inf`` positions are excluded, finite
values are added to the logits before the softmax). The effective ``combined_mask`` is the conjunction
``attn_mask & causal_{S-L+l} & expand(plan)``.

**Head mapping**

Selection heads :math:`H_{sel} \in \{1, H_{kv}, H_{q}\}` with :math:`H_{q} \bmod H_{sel} == 0`. Query head
:math:`h` maps to selection head :math:`sh = \lfloor h / (H_{q} / H_{sel}) \rfloor`. When :math:`H_{sel}=1`
(QSA), all query heads share the single selection; when :math:`H_{sel}=H_{kv}`, each group of
:math:`H_{q} / H_{kv}` query heads shares a per-KV-head selection; when :math:`H_{sel}=H_{q}`, every query head
carries its own selection.

**Decode incremental contract**

The op is *stateless with respect to the selection*: it recomputes attention over the selected set each step. The
incrementality burden is entirely on the upstream producer (which maintains the block-summary cache). For a decode
step (:math:`B_q=1`) with :math:`B` selected blocks per query and :math:`H_{kv}` KV heads, the per-query work is
:math:`O(B \cdot B_k \cdot D_v)` for the value gather and :math:`O(B \cdot B_k \cdot D_h)` for scoring —
independent of total history length :math:`S`. This is what makes long-context decode tractable: the op's cost is
bounded by the budget, not the sequence length.

**Relationship to ``SparsePA``**

``SparseSDPA`` and ``SparsePA`` share the exact same mathematical kernel (score :math:`\to` online-softmax
:math:`\to` weighted-gather) and the same clean selection contract. The only difference is the KV memory contract:

- ``SparseSDPA``: dense / KVCache-managed contiguous tensors, one physical page per sequence, identity block
  table. Serves as the reference / decomposition form for validation and early prototyping.
- ``SparsePA``: physically paged ``[num_blocks, Hkv, block_size, D]`` cache with an explicit
  ``block_indices`` / ``block_indices_begins`` table. This is the production paged execution route.

A lowering pass may convert ``SparseSDPA`` into ``SparsePA`` (or vice versa) by synthesizing a degenerate / real
page table; the fused GPU primitive dispatch is shared.

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def sparse_sdpa(query, key, value,
                    sel_indices, sel_count,
                    attn_mask=None, scale=None,
                    *, num_heads, num_kv_heads, h_sel, k_head_size, v_head_size,
                    sel_block_size, q_block_size=1, causal=True):
        # query:             [B, Hq, L, Dh]  (already projected/RMSNorm'ed/RoPE'ed)
        # key:               [B, Hkv, S, Dh]
        # value:             [B, Hkv, S, Dv]
        # sel_indices:       [B, H_sel, num_q_blocks, K_max]   (int32; strictly ascending, -1 padded past count)
        # sel_count:         [B, H_sel, num_q_blocks]           (int32; valid block count)
        # attn_mask:         broadcastable to [B, 1, L, S] optional (SDPA semantics; boolean or float-additive)
        # scale:             scalar optional; defaults to 1/sqrt(Dh)
        B, Hq, L, Dh = query.shape
        Hkv = num_kv_heads
        S = key.shape[2]
        Dv = v_head_size
        Bk = sel_block_size
        Bq = q_block_size
        sc = scale if scale is not None else 1.0 / sqrt(Dh)
        num_q_blocks = ceil(L / Bq)                 # first selection dimension (== L when Bq == 1)
        q_per_sel = Hq // h_sel                     # query heads per selection head (Hq % h_sel == 0)

        q = query                                            # [B, Hq, L, Dh]
        out = zeros((B, Hq, L, Dv))

        for b in range(B):
            for qb in range(num_q_blocks):
                l0 = qb * Bq
                l1 = min(l0 + Bq, L)                          # query block row range [l0, l1)
                for sh in range(h_sel):
                    nblocks = sel_count[b, sh, qb]
                    idx = sel_indices[b, sh, qb, :nblocks]    # [nblocks] logical block indices (ascending set)
                    # Materialize the absolute KV position of every gathered token, then truncate any
                    # out-of-range (kv_abs_pos >= S) entries introduced by a partial final block.
                    blk_base = idx * Bk                        # absolute KV start of each selected block
                    kpos = concat([blk_base[:, None] + arange(Bk)[None, :]], axis=-1).reshape(-1)
                    valid = kpos < S                           # truncate partial blocks that exceed the cache
                    kpos = kpos[valid]
                    K_g = gather_blocks(key[b], idx, Bk)[valid]   # [n_tok, Hkv, Dh]
                    V_g = gather_blocks(value[b], idx, Bk)[valid] # [n_tok, Hkv, Dv]
                    for h in range(q_per_sel * sh, q_per_sel * (sh + 1)):
                        kv = h // (Hq // Hkv)                  # KV head of query head h
                        # scores over the selected set. The query's absolute position is right-aligned:
                        # q_abs = S - L + l (generative / transformers convention); causal keeps kpos <= q_abs.
                        scores = np.sum(q[b, h, l0:l1, None, :] * K_g[None, :, kv, :], axis=-1) * sc
                        # Build the effective combined_mask = attn_mask & causal & expand(plan) as a boolean.
                        eff = ones((l1 - l0, kpos.shape[0]), dtype=bool)   # within selected blocks (plan)
                        if causal:
                            q_abs = S - L + (l0 + arange(l1 - l0))         # [Lq] absolute query positions
                            eff = eff & (q_abs[:, None] >= kpos[None, :])  # bottom-right aligned, self-inclusive
                        if attn_mask is not None:
                            if attn_mask.dtype == bool:
                                eff = eff & attn_mask[b, 0, l0:l1, kpos]
                            else:
                                m = attn_mask[b, 0, l0:l1, kpos]
                                eff = eff & (m != -inf)
                                scores = scores + where(m != -inf, m, 0)
                        # Apply the mask: excluded positions get -inf (or NaN guard). Fully-masked rows -> 0.
                        scores = where(eff, scores, -inf)
                        all_masked = ~np.any(eff, axis=-1)                 # rows with no valid key
                        probs = softmax(scores, axis=-1)                   # online, over valid tokens only
                        probs = where(all_masked[:, None], zeros_like(probs), probs)  # NaN guard
                        out[b, h, l0:l1] = sum(probs[:, :, None] * V_g[None, :, kv, :], axis=1)

        return out                                           # [B, Hq, L, Dv]


**Attributes**

* *num_heads*

  * **Description**: Number of query heads :math:`H_q`.
  * **Type**: ``int``
  * **Required**: *yes*

* *num_kv_heads*

  * **Description**: Number of KV heads :math:`H_{kv}`. Must divide ``num_heads``.
  * **Type**: ``int``
  * **Required**: *yes*

* *h_sel*

  * **Description**: Number of selection heads :math:`H_{sel}`, matching the producer. Must be ``1``,
    ``num_kv_heads``, or ``num_heads``.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``h_sel`` is ``1``, ``num_kv_heads``, or ``num_heads``; ``num_heads % h_sel == 0``.

* *k_head_size*

  * **Description**: Head dimension :math:`D_h` of the keys.
  * **Type**: ``int``
  * **Required**: *yes*

* *v_head_size*

  * **Description**: Head dimension :math:`D_v` of the values.
  * **Type**: ``int``
  * **Required**: *yes*

* *sel_block_size*

  * **Description**: Number of tokens per selected KV block, :math:`B_k`, matching the producer's
    ``compress_ratio``. The consumer gathers full blocks of this size; the causal-diagonal block may be
    partial, bounded by the query position inclusive.
  * **Type**: ``int``
  * **Required**: *yes*

* *q_block_size*

  * **Description**: Number of query tokens sharing a single selection, :math:`B_q`. Default ``1`` (decode /
    per-token selection). When :math:`B_q > 1` (e.g. FlashVSR block sparse with :math:`B_q = 128`),
    ``sel_indices``/``sel_count`` carry one selection per query block (first selection dimension
    ``num_q_blocks = ceil(L / B_q)``) and it is replicated across the :math:`B_q` rows. When :math:`B_q > 1`,
    the history length must align: ``(S - L) % q_block_size == 0`` (so every query block maps to a whole number
    of :math:`B_q`-aligned KV spans).
  * **Type**: ``int``
  * **Default value**: ``1``
  * **Required**: *no*

* *causal*

  * **Description**: Whether the causal-diagonal block is gathered as a partial block whose valid keys satisfy
    :math:`kpos \le q_{abs}` (inclusive, self-attended), where :math:`q_{abs} = S - L + l` is the query's
    **bottom-right aligned** absolute position (the generative / transformers convention). When ``false``, all
    tokens of every selected block are attended.
  * **Type**: ``bool``
  * **Default value**: ``true``
  * **Required**: *no*


**Inputs**

* **0**: ``query``
  A 4D tensor of type *T* with shape ``[B, Hq, L, Dh]``.
  Query hidden states, already projected, RMSNorm'ed and RoPE'ed by the attention layer. **Required.**

* **1**: ``key``
  A 4D tensor of type *T* with shape ``[B, Hkv, S, Dh]``.
  Dense key history of length :math:`S`, from the ``KVCache`` "present" output. **Required.**

* **2**: ``value``
  A 4D tensor of type *T* with shape ``[B, Hkv, S, Dv]``.
  Dense value history of length :math:`S`. **Required.**

* **3**: ``sel_indices``
  A 4D tensor of type *T_IND* with shape ``[B, H_sel, num_q_blocks, K_max]``.
  Per-query-block, per-selection-head selected *block* indices emitted by the upstream producer, where
  ``num_q_blocks = ceil(L / B_q)`` (equals ``L`` when ``B_q == 1``). ``-1`` padded. **Required**, unless the plan
  is instead carried directly by ``attn_mask`` (see "Mask-only plans for unbounded-K_max producers" above),
  in which case this input is omitted.

* **4**: ``sel_count``
  A 3D tensor of type *T_IND* with shape ``[B, H_sel, num_q_blocks]``.
  Valid selected block count per query block per selection head. **Required, unless omitted together with**
  ``sel_indices`` **when the plan is carried by** ``attn_mask``.

* **5**: ``attn_mask``
  A tensor broadcastable to ``[B, 1, L, S]`` (optional, unless it is carrying the sparsity plan — see below).
  Attention mask in SDPA-13 semantics: either a **boolean** mask (``False`` excludes a position) or a
  **floating-point additive** mask (``-inf`` excludes, finite values are added to the logits). The effective
  mask is the conjunction ``attn_mask & causal & expand(plan)``. When omitted, only ``causal & expand(plan)``
  applies. **Optional**, except when ``sel_indices``/``sel_count`` are omitted: for producers with no fixed
  per-row top-:math:`K` (e.g. XAttention's threshold-based block selection), the full sparsity plan may instead
  be expanded directly into ``attn_mask`` (see "Mask-only plans for unbounded-K_max producers" above), in
  which case ``attn_mask`` becomes **Required**.

* **6**: ``scale``
  A scalar tensor of type *T* (optional).
  Attention scale. When omitted, ``1.0 / sqrt(k_head_size)`` is used. **Optional.**


**Outputs**

* **0**: ``output``
  A 4D tensor of type *T* with shape ``[B, Hq, L, Dv]``.
  The sparse attention output, in the same layout as ``query``.


**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` or ``int64``.
* ``num_heads % num_kv_heads == 0``; the group size ``num_heads // num_kv_heads`` must be positive.
* ``h_sel`` is ``1``, ``num_kv_heads``, or ``num_heads``; ``num_heads % h_sel == 0``.
* ``key``/``value`` second dimension equals ``num_kv_heads``.
* ``query`` first dimension equals ``key`` first dimension (:math:`B`); ``query`` sequence dimension :math:`L`
  is related to the last two selection dims by ``sel_indices`` dim 2 ``== ceil(L / q_block_size)`` and
  ``sel_count`` dim 2 ``== ceil(L / q_block_size)``. When ``q_block_size == 1``, this equals ``L``.
* ``sel_indices`` last dimension ``K_max`` is **determined by the producer**: ``block_topk + 1`` for a QSA-style
  producer, or ``ceil(S / sel_block_size)`` when the selection is produced from a block-level mask conversion
  (the full set of logical KV blocks). ``sel_count[b, sh, qb] <= K_max``, and the valid prefix
  ``sel_indices[b, sh, qb, :sel_count[b, sh, qb]]`` is **strictly ascending** and contains no ``-1`` entries.
  When a producer has no bound on ``K_max`` tighter than ``ceil(S / sel_block_size)`` (e.g. XAttention's
  threshold-based selection), ``sel_indices``/``sel_count`` should be omitted and the plan carried directly via
  ``attn_mask`` instead — see "Mask-only plans for unbounded-K_max producers".
* Exactly one plan representation is present: either both ``sel_indices`` and ``sel_count``, or a plan-carrying
  ``attn_mask`` with ``sel_indices``/``sel_count`` both omitted.
* When ``q_block_size > 1``, the history length satisfies ``(S - L) % q_block_size == 0``.
* The selected blocks form a **strictly ascending, duplicate-free** sequence; the consumer gathers exactly the
  valid prefix.
* When ``attn_mask`` is present it must be broadcastable to ``[B, 1, L, S]`` and is boolean or float-additive
  per SDPA-13.
* When ``causal`` is true, the causal-diagonal block is read only up to the query's right-aligned absolute
  position inclusive (:math:`kpos \le q_{abs} = S - L + l`).
* Gathered tokens with absolute position ``>= S`` (introduced by a partial final block) are truncated out.
* When a row's selection is empty (``sel_count == 0``), the op produces a zero output vector for that row.


**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Examples**

**Example 1 — Qwen4 Token Sparse** (:math:`B_q = 1`, :math:`B_k = 4`, :math:`H_{sel} = 1`).

The producer emits one selection per query *token* (:math:`B_q = 1`, so ``num_q_blocks == L``). ``sel_indices``
encodes the top-:math:`K` complete blocks (from the per-head ReLU score) plus the causal diagonal. With
``B=1``, ``L=1``, dense key history ``S=10``, ``sel_block_size=4``, ``block_topk=2`` (``K_max=3``): the query's
absolute position is :math:`q_{abs} = S - L + l = 10 - 1 + 0 = 9`; causally-valid complete blocks are those with
``b < (9+1)//4 = 2``, i.e. ``{0, 1}``. After a stable top-2 the plan is ``sel_indices = [[[[0, 1, 2]]]]``,
``sel_count = [[[3]]]``. Block ``2`` covers KV positions ``[8..11]``; it is truncated to ``< S = 10``
(positions ``{8, 9}``) and then, under ``causal = true``, further to :math:`kpos \le q_{abs} = 9`, so the gathered
valid keys of the diagonal block are ``{8, 9}``.

.. code-block:: xml
   :force:

   <layer ... type="SparseSDPA">
       <data num_heads="24" num_kv_heads="2" h_sel="1" k_head_size="256" v_head_size="256"
             sel_block_size="4" q_block_size="1" causal="true"/>
       <input>
           <port id="0">   <!-- query: [B, Hq, L, Dh] -->
               <dim>1</dim><dim>24</dim><dim>1</dim><dim>256</dim>
           </port>
           <port id="1">   <!-- key: [B, Hkv, S, Dh] -->
               <dim>1</dim><dim>2</dim><dim>10</dim><dim>256</dim>
           </port>
           <port id="2">   <!-- value: [B, Hkv, S, Dv] -->
               <dim>1</dim><dim>2</dim><dim>10</dim><dim>256</dim>
           </port>
           <port id="3">   <!-- sel_indices: [B, H_sel, num_q_blocks, K_max] (num_q_blocks = ceil(L/Bq) = 1) -->
               <dim>1</dim><dim>1</dim><dim>1</dim><dim>3</dim>
           </port>
           <port id="4">   <!-- sel_count: [B, H_sel, num_q_blocks] -->
               <dim>1</dim><dim>1</dim><dim>1</dim>
           </port>
       </input>
       <output>
           <port id="5">   <!-- output: [B, Hq, L, Dv] -->
               <dim>1</dim><dim>24</dim><dim>1</dim><dim>256</dim>
           </port>
       </output>
   </layer>

**Example 2 — FlashVSR Block Sparse** (:math:`B_q = 128`, :math:`B_k = 128`, :math:`H_{sel} = H_q = H_{kv}`,
non-causal).

FlashVSR's temporal block sparsity is expressed as a block-level sparse mask carried once per :math:`128`-token
query block, using **multi-head attention** (:math:`H_q = H_{kv} = H_{sel}`). With ``B=1``, ``L=512``,
:math:`B_q = 128`, the first selection dimension is ``num_q_blocks = ceil(512 / 128) = 4``. Each of the 4 query
blocks carries its own :math:`H_{sel} = H_q` selection over the :math:`S`-token KV history. A block-level sparse
mask that keeps, say, blocks ``{0, 2, 3}`` for every selection head of query block ``qb=0`` is encoded as
``sel_indices[0, :, 0, :] = [[0, 2, 3, -1, ...], ...]`` and ``sel_count[0, :, 0] = [3, 3, ...]`` (replicated across
the :math:`H_{sel}` selection heads). With :math:`B_k = 128` and :math:`S = 512` there are ``S / B_k = 4`` logical
blocks; the selection covers :math:`3` of them. Non-causal (:math:`causal = \mathrm{false}`), so all tokens of
every selected block are attended (no causal-diagonal block is needed). **The input tokens must be pre-permuted
into window-major order** (FlashVSR ``WindowPartition3D``) so that each :math:`B_q`/:math:`B_k` block is exactly
one 3D window.

.. code-block:: xml
   :force:

   <layer ... type="SparseSDPA">
       <data num_heads="12" num_kv_heads="12" h_sel="12" k_head_size="128" v_head_size="128"
             sel_block_size="128" q_block_size="128" causal="false"/>
       <input>
           <port id="0">   <!-- query: [B, Hq, L, Dh] -->
               <dim>1</dim><dim>12</dim><dim>512</dim><dim>128</dim>
           </port>
           <port id="1">   <!-- key: [B, Hkv, S, Dh] -->
               <dim>1</dim><dim>12</dim><dim>512</dim><dim>128</dim>
           </port>
           <port id="2">   <!-- value: [B, Hkv, S, Dv] -->
               <dim>1</dim><dim>12</dim><dim>512</dim><dim>128</dim>
           </port>
           <port id="3">   <!-- sel_indices: [B, H_sel, num_q_blocks, K_max] (num_q_blocks = ceil(512/128) = 4) -->
               <dim>1</dim><dim>12</dim><dim>4</dim><dim>4</dim>
           </port>
           <port id="4">   <!-- sel_count: [B, H_sel, num_q_blocks] -->
               <dim>1</dim><dim>12</dim><dim>4</dim>
           </port>
       </input>
       <output>
           <port id="5">   <!-- output: [B, Hq, L, Dv] -->
               <dim>1</dim><dim>12</dim><dim>512</dim><dim>128</dim>
           </port>
       </output>
   </layer>

**Example 3 — XAttention Threshold-Based Sparsity** (:math:`B_q = B_k = 128`, plan carried by ``attn_mask``,
``sel_indices``/``sel_count`` omitted).

XAttention's block selection accumulates the highest-scoring blocks per query block until their cumulative
softmax mass crosses a threshold fraction, so the selected-block count is content-dependent with no fixed cap.
With ``B=1``, ``L=512``, ``S=512``, :math:`B_q = B_k = 128` (``num_q_blocks = num_k_blocks = 4``), suppose query
block ``qb=1`` keeps blocks ``{0, 1}`` while query block ``qb=3`` (near-uniform scores) keeps all of
``{0, 1, 2, 3}``: a fixed-width ``sel_indices`` would need ``K_max = 4 == num_k_blocks`` to cover the worst case,
buying no index-list savings for ``qb=3``. The plan is instead expanded to a per-token boolean mask (every token
of a selected block is ``True``) and passed as ``attn_mask``; ``sel_indices`` and ``sel_count`` are omitted.

.. code-block:: xml
   :force:

   <layer ... type="SparseSDPA">
       <data num_heads="12" num_kv_heads="12" h_sel="12" k_head_size="128" v_head_size="128"
             sel_block_size="128" q_block_size="128" causal="false"/>
       <input>
           <port id="0">   <!-- query: [B, Hq, L, Dh] -->
               <dim>1</dim><dim>12</dim><dim>512</dim><dim>128</dim>
           </port>
           <port id="1">   <!-- key: [B, Hkv, S, Dh] -->
               <dim>1</dim><dim>12</dim><dim>512</dim><dim>128</dim>
           </port>
           <port id="2">   <!-- value: [B, Hkv, S, Dv] -->
               <dim>1</dim><dim>12</dim><dim>512</dim><dim>128</dim>
           </port>
           <port id="3">   <!-- attn_mask: [B, H_sel, L, S], carries the plan directly; sel_indices/sel_count omitted -->
               <dim>1</dim><dim>12</dim><dim>512</dim><dim>512</dim>
           </port>
       </input>
       <output>
           <port id="4">   <!-- output: [B, Hq, L, Dv] -->
               <dim>1</dim><dim>12</dim><dim>512</dim><dim>128</dim>
           </port>
       </output>
   </layer>
