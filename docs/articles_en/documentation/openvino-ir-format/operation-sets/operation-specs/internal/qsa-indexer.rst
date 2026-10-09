.. {#openvino_docs_ops_internal_QSAIndexer}

QSAIndexer
==========

.. meta::
  :description: Learn about QSAIndexer - the standard, stateful, functional-interface block-level indexer for Qwen Sparse Attention (QSA). It consumes externally projected query and key tokens plus an external ReadValue/Assign-wired state, maintains an incremental pooled-key summary cache, and emits a clean (sel_indices, sel_count) block-selection contract for the unified SparseSDPA / SparsePA consumers.

**Versioned name**: *QSAIndexer*

**Category**: *Internal*

**Short description**:
The *QSAIndexer* is the **model-specific plan producer** for Qwen Sparse Attention (QSA) in its
**standard, non-paged, stateful** form. It is a pure *Functional State Interface* operation: all recurrent
state (the completed-block summary cache and the incomplete pending block) is passed **in and out** as explicit
state tensors, which the surrounding graph wires through external ``ReadValue`` / ``Assign`` nodes (optionally
through a ``Gather(beam_idx)`` for beam search). This makes it suitable for **single-sequence / fixed-batch**
SDPA pipelines (e.g. the ``LLMInferenceSDPAModule`` in ``openvino.pipeline.mx``) that do not use a paged KV
cache. For continuous batching over a PagedAttention page table, see the separate
:doc:`PagedQSAIndexer <paged-qsa-indexer>` operation.

**Functional State Interface and graph wiring**

Unlike the paged variant, *QSAIndexer* owns **no internal, opaque state** and reads **no page table**. It is a
stateless function of its inputs plus the explicit state passed in on ``summary_past`` / ``indexer_k_past`` /
``indexer_k_start_past``, and it returns the *updated* state on ``summary_present`` / ``indexer_k_present`` /
``indexer_k_start_present``. The surrounding graph is responsible for the state lifecycle:

.. code-block:: text

   ReadValue(summary_state)          --> summary_past         --> QSAIndexer --> summary_present         --> Assign(summary_state)
   ReadValue(indexer_k_state)        --> indexer_k_past       --> QSAIndexer --> indexer_k_present       --> Assign(indexer_k_state)
   ReadValue(indexer_k_start_state)  --> indexer_k_start_past --> QSAIndexer --> indexer_k_start_present --> Assign(indexer_k_start_state)

   Gather(ReadValue(summary_state), beam_idx, axis=0)          --> summary_past
   Gather(ReadValue(indexer_k_state), beam_idx, axis=0)        --> indexer_k_past
   Gather(ReadValue(indexer_k_start_state), beam_idx, axis=0)  --> indexer_k_start_past

   beam_idx (a Parameter) -> Gather(ReadValue(var), beam_idx, axis=0)

When beam search is used, the **state tensors only** are gathered with ``beam_idx`` before the op, so each
hypothesis branch carries its own independent summary / pending / RoPE-start state. ``beam_idx`` is a plain
``Parameter``; the ``Gather(ReadValue(var), beam_idx, axis=0)`` applies **only** to the three state tensors
(``summary_past``, ``indexer_k_past``, ``indexer_k_start_past``) — never to ``q_idx``, ``k_idx``, or
``k_rope_cos_sin``, which are produced fresh for the step and are identical across hypotheses. The op itself is
pure: the same inputs yield the same outputs, making it trivially re-runnable for rollback and deterministic
across devices.

**Detailed description**

The *QSAIndexer* realizes the *producer* side of the sparse-attention split in its **standard, non-paged**
form. It is *model-specific*: it encodes the exact QSA reference math (mean-pool :math:`\to` RMSNorm :math:`\to`
block-start RoPE :math:`\to` per-head ReLU score :math:`\to` top-k). The indexer GEMM weight (``index_qk_proj``)
is **kept outside** this op as a standard ``FullyConnected`` / ``MatMul`` for four reasons:

1. **Quantization (NNCF INT4/FP8).** Keeping the projection as an ordinary MatMul lets the NNCF weight-
   compression passes rewrite it without touching an opaque fused primitive. Dequantizing a compressed
   projection to BF16 for an in-op GEMM costs orders of magnitude more DRAM traffic than the <0.1% decode
   traffic the fusion would save.
2. **Horizontal FC fusion.** A standalone projection can be fused horizontally with the main
   ``q_proj``/``k_proj``/``v_proj`` for the shared input, a transform impossible when the GEMM is embedded in
   the indexer primitive.
3. **Tensor parallelism (TP).** A standalone MatMul can be sharded across devices by head / dimension without
   fighting a fused kernel's fixed input layout.
4. **Kernel combinatorics.** Fusing the projection into the indexer multiplies the kernel family by every
   precision / layout combination; keeping it a standard FC keeps the kernel space small.

The *K-side compression chain*, by contrast, **must** stay inside the indexer semantics because it is
*non-associative* and *incremental*: :math:`\mathrm{RMSNorm}(\mathrm{Mean}(K)) \ne \mathrm{Mean}(\mathrm{RMSNorm}(K))`,
and a block summary can only be maintained :math:`O(1)` per new token by an internal cache. Weight-free-ness
therefore applies to *weights* (explicit, external GEMM) and not to *compression semantics* (atomic, cached,
inside the op).

**Head layout and the index-head grouping**

The indexer distinguishes two head dimensions that are **derived from the input shapes**, not from an attribute:

- ``q_idx`` carries ``H_iq`` index-*query* heads in its penultimate dimension (``[B, L, H_iq, D_idx]``).
- ``k_idx`` carries ``H_ik`` index-*key* heads in its penultimate dimension (``[B, L, H_ik, D_idx]``).

``H_ik`` is configurable and taken from the ``k_idx`` shape, never hard-coded to ``1``. This lets a single block
summary key be stored per index-key head, so models that need more than one key projection head are expressible
without adding a special-case attribute. The score for selection head ``sh`` is a grouped, per-query-head ReLU
reduction. The grouping is fully determined by the head counts:

.. math::

    H_{iq} \bmod H_{ik} = 0, \quad G = H_{iq} / H_{ik}, \quad \mathrm{kv}(h) = \lfloor h / G \rfloor, \quad
    \mathrm{sel}(h) = \lfloor \mathrm{kv}(h) \cdot H_{sel} / H_{ik} \rfloor

    s_{b, sh} = \frac{1}{\sqrt{D_{idx}}} \sum_{\substack{h \in [0, H_{iq}) \\ \mathrm{sel}(h) = sh}}
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
per index-query head before reduction over heads, matching the reference QSA implementation. Under QSA
(:math:`H_{iq} = 4`, :math:`H_{ik} = 1`, :math:`H_{sel} = 1`) all four index-query heads map to the single
index-key head and the single selection head, so the score is the sum over all four per-head ReLU dot-products.

**State organization**

The K-side history is maintained as two explicit state tensors plus one RoPE-start state:

- ``summary_past`` / ``summary_present`` ``[B, H_ik, N_c, D_idx]``: the mean-pooled, RMSNorm'ed, block-start-
  RoPE'ed summary keys of the **completed** blocks (:math:`N_c = \lfloor (S_0 + L) / r \rfloor` complete blocks
  after this step, with :math:`S_0` the number of already-stored token positions). This is the incrementally-
  updated artifact that makes the per-token update :math:`O(1)`.
- ``indexer_k_past`` / ``indexer_k_present`` ``[B, H_ik, P, D_idx]``: the projected, per-token indexer keys of the
  **currently incomplete** block, with :math:`P = S_0 \bmod r \in [0, r)` on input (``indexer_k_past``) and
  :math:`P' = (S_0 + L) \bmod r` on output (``indexer_k_present``). These are the raw
  (unrotated) keys that will be pooled into the next completed-block summary when the block fills.
- ``indexer_k_start_past`` / ``indexer_k_start_present`` ``[B, 2 * indexer_rotary_dim]``: the precomputed RoPE
  ``(cos, sin)`` at the block-start position of the pending block, refreshed when the first token of a block
  arrives.

The **logical position** of a token is ``S_0 + l`` (batch element ``b``, step token ``l``), where ``S_0`` is the
number of already-stored key positions for that batch element before this step (equivalently, the length of the
history already summarized into ``summary_past``/``indexer_k_past``). A block is *completed* once exactly :math:`r`
raw keys have been appended; the last, partially-filled block is **not** summarized but emitted as a **valid
block index** in the output (the producer-completeness rule).

**Score computation**

For a query at logical position :math:`p` with projected query :math:`q_h \in \mathbb{R}^{D_{idx}}` (per index-query
head :math:`h`) and block :math:`b` whose per-index-key-head summary key is
:math:`\bar{k}_{b, \mathrm{kv}(h)} \in \mathbb{R}^{D_{idx}}`, the per-selection-head score is the grouped reduction
over index-query heads (see the formula above), mapping each query head to its selection head via
:math:`\mathrm{sel}(h)`. The ReLU is applied *per index-query head before reduction over heads*, matching the
reference QSA implementation. The indexer query :math:`q` arrives already projected (external GEMM) and already
RMSNorm'ed and RoPE'ed; the op does **not** apply RoPE or RMSNorm to the query (those are owned by the surrounding
attention layer / projection). Only the pooled block key undergoes RMSNorm (via ``k_norm_weight``) and block-start
RoPE (via the ``k_rope_cos_sin`` state).

The RMSNorm used for the pooled block key is defined as

.. math::

    \mathrm{rms\_norm}(x, w, \mathrm{eps}) = x \cdot \mathrm{rsqrt}\!\left( \mathrm{mean}(x^2) + \mathrm{eps} \right) \cdot w

computed in ``fp32`` regardless of the input precision. Qwen's Gemma-style scale :math:`(1 + w)` is folded by the
model converter into the stored weight :math:`w' = 1 + w`, so the op consumes the already-shifted weight.

**Selection, tie-breaking, and top-k**

- A block is *causally valid* for a query at logical position :math:`p` iff its last token is at most :math:`p`,
  i.e. :math:`N_{complete}(p) = \lfloor (p + 1) / r \rfloor` blocks are eligible. Only these causally-valid
  complete blocks participate in the top-k (future / not-yet-written blocks never leak into the selection).
- The number of selected *complete* blocks per selection head is capped at ``block_topk``; the output width is
  ``K_max = block_topk + 1`` (the extra slot holds the causal-diagonal / incomplete block, when present).
- **Tie-breaking and ordering**: among equal scores the *smaller block index* (earlier block) wins; the op
  computes the selection as ``argsort(-scores, kind='stable')``, takes the first ``k`` entries, and emits them in
  **strictly ascending** block order (``sort(order[:k])``), yielding a deterministic, implementation-independent,
  monotone result. A plugin may rely on ``sel_indices[t, sh, :sel_count]`` being sorted ascending.
- **Causal-diagonal inclusion**: when the query at logical position :math:`p` lies inside an incomplete block
  (i.e. :math:`(p+1) \bmod r \neq 0`), the diagonal block with logical index :math:`\lfloor (p+1)/r \rfloor` is
  written **immediately after** the selected complete blocks, at ``sel_indices[..., k]``, and
  ``sel_count = k + 1``. Writing it there (rather than a fixed slot) guarantees the valid prefix is contiguous
  with no interior ``-1`` holes.

**Block grid and causal rules**

The KV history of a batch element is partitioned into blocks of :math:`r` tokens. Block :math:`b` covers tokens
:math:`[b \cdot r, (b+1) \cdot r - 1]`. Scoring is causal: a query at logical position :math:`p` may only select
complete blocks with :math:`b \le \lfloor (p+1)/r \rfloor - 1`; the current incomplete block (the causal
diagonal) is always appended separately. There is no ``is_causal`` attribute; the exact per-token causal
truncation is applied by the consumers (``SparseSDPA`` / ``SparsePA``) via the **token-level fine filter**
:math:`kpos \le q_{abs}` (see the consumer specs).

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def qsa_indexer(q_idx, k_idx, k_rope_cos_sin, summary_past, indexer_k_past, indexer_k_start_past,
                    k_norm_weight,
                    *, compress_ratio, block_topk, h_sel, indexer_rotary_dim,
                    indexer_head_dim, eps, scale):
        # q_idx:                 [B, L, H_iq, D_idx]          (projected, RMSNorm'ed and RoPE'ed externally)
        # k_idx:                 [B, L, H_ik, D_idx]          (projected by external GEMM)
        # k_rope_cos_sin:        [B, L, 2 * indexer_rotary_dim]  (key-side RoPE (cos,sin); block-start phase)
        # summary_past:          [B, H_ik, N_c, D_idx]        (completed-block summaries; N_c dynamic)
        # indexer_k_past:        [B, H_ik, P, D_idx]          (incomplete pending keys; 0 <= P < r)
        # indexer_k_start_past:  [B, 2 * indexer_rotary_dim]  (RoPE (cos,sin) at the pending block start)
        # k_norm_weight:         [D_idx]
        # returns: sel_indices               [B, H_sel, L, K_max]          (int32; -1 = pad), K_max = block_topk + 1
        #          sel_count                 [B, H_sel, L]                (int32)
        #          summary_present           [B, H_ik, N_c', D_idx]
        #          indexer_k_present         [B, H_ik, P', D_idx]
        #          indexer_k_start_present   [B, 2 * indexer_rotary_dim]
        B, L, H_iq, Di = q_idx.shape
        Hik = k_idx.shape[2]                     # index-key heads, taken from the shape (configurable!)
        Hsel = h_sel
        r = compress_ratio
        Kmax = block_topk + 1                    # +1 slot for the causal-diagonal / incomplete block
        G = H_iq // Hik                          # index-query heads per index-key head (H_iq % Hik == 0)
        assert H_iq % Hik == 0 and Hsel in (1, Hik) and r >= 1
        S0 = summary_past.shape[2] * r + indexer_k_past.shape[2]   # already-stored logical length

        # 1. Append current keys to the pending block; flush completed blocks into summary_present.
        #    k_t: [B, H_ik, L, D_idx] (transpose of [B, L, H_ik, D_idx]).
        k_t = transpose(k_idx, (0, 2, 1, 3))                       # [B, Hik, L, Di]
        P = indexer_k_past.shape[2]                # number of pending keys carried into this step
        pending = concat([indexer_k_past, k_t], axis=2)              # [B, Hik, P+L, Di]
        n_pending = pending.shape[2]
        n_complete = n_pending // r              # blocks fully filled in this step
        rem = n_pending % r
        summary = concat([summary_past,
                          zeros((B, Hik, n_complete, Di))], axis=2)   # [B, Hik, N_c+n_complete, Di]
        for b_idx in range(n_complete):
            b = summary_past.shape[2] + b_idx    # logical block index
            grp = pending[:, :, b_idx*r : (b_idx+1)*r, :]           # [B, Hik, r, Di]
            pooled = mean(grp.astype(float32), axis=2).astype(k_idx.dtype)   # [B, Hik, Di]
            nb = rms_norm(pooled, k_norm_weight, eps)              # [B, Hik, Di]
            # block-start RoPE phase of this completed block: its first token is at logical offset
            # j0 = b_idx*r - P within this step (negative => it began before this step, use the pending phase).
            j0 = b_idx * r - P
            cs = indexer_k_start_past if j0 < 0 else k_rope_cos_sin[:, j0]   # [B, 2*rot]
            rot = nb[..., :indexer_rotary_dim]
            bar = concat([rope(rot, cs), nb[..., indexer_rotary_dim:]], axis=-1)  # [B, Hik, Di]
            summary[:, :, summary_past.shape[2] + b_idx, :] = bar
        indexer_k_present = pending[:, :, n_complete*r :, :]         # [B, Hik, rem, Di]
        summary_present = summary
        # indexer_k_start_present: the RoPE phase of the next (new) block start.
        # rem == 0: no pending block remains; the next step's first token starts a new block and refreshes the
        # phase from its own k_rope_cos_sin row, so the carried value is unused.
        if rem == 0:
            indexer_k_start_present = indexer_k_start_past
        else:
            j0 = n_complete * r - P          # offset in this step of the first token of the still-open block
            indexer_k_start_present = indexer_k_start_past if j0 < 0 else k_rope_cos_sin[:, j0]

        # 2. Score the causally-valid complete blocks per selection head (grouped per-query-head reduction).
        sel_indices = full((B, Hsel, L, Kmax), -1, dtype=int32)
        sel_count   = zeros((B, Hsel, L), dtype=int32)
        for b_batch in range(B):
            for l in range(L):
                pos = S0 + l                              # logical position of step token l
                n_c = (pos + 1) // r                      # causally-valid complete blocks for this query
                for sh in range(Hsel):
                    k = min(block_topk, n_c)
                    # For each index-key head kv, map it to selection head sel = kv*Hsel//Hik and accumulate
                    # only the query heads that fall in that selection head's group.
                    scores = np.zeros(n_c)
                    for kv in range(Hik):
                        if (kv * Hsel) // Hik != sh:
                            continue
                        # index-query heads h in [kv*G, (kv+1)*G) share index-key head kv.
                        q_h = q_idx[b_batch, l, kv*G:(kv+1)*G, :]               # [G, Di]
                        bar = summary_present[b_batch, kv, :n_c, :]              # [n_c, Di]
                        # ReLU per index-query head on the dot product, then sum over heads (reference: relu(q·k).sum(heads))
                        scores += sum(relu(sum(q_h[None, :, :] * bar[:, None, :], axis=-1)), axis=1)  # [n_c]
                    scores = scores * scale
                    order = np.argsort(-scores, kind='stable')
                    picked = sort(order[:k])              # strictly ascending block indices
                    sel_indices[b_batch, sh, l, :k] = picked
                    sel_count[b_batch, sh, l] = k
                    if (pos + 1) % r != 0:
                        inc = (pos + 1) // r              # causal-diagonal / incomplete block (> all complete)
                        sel_indices[b_batch, sh, l, k] = inc
                        sel_count[b_batch, sh, l] = k + 1
        return sel_indices, sel_count, summary_present, indexer_k_present, indexer_k_start_present


**Attributes**

* *compress_ratio*

  * **Description**: :math:`r`, the number of raw token keys pooled into one block key.
  * **Type**: ``int``
  * **Default value**: ``4``
  * **Required**: *no*
  * **Constraints**: ``compress_ratio >= 1``.

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
    ``indexer_head_dim - indexer_rotary_dim`` dims are passed through unrotated. Must satisfy
    ``indexer_rotary_dim <= indexer_head_dim``.
  * **Type**: ``int``
  * **Required**: *yes*

* *indexer_head_dim*

  * **Description**: Head dimension :math:`D_{idx}` of the indexer.
  * **Type**: ``int``
  * **Required**: *yes*

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
  A 4D tensor of type *T* with shape ``[B, L, H_iq, D_idx]``.
  Projected indexer queries emitted by the external ``FullyConnected`` / ``MatMul`` (``index_qk_proj``), already
  RMSNorm'ed and RoPE'ed (the q-side norm/RoPE live outside the op). **Required.**

* **1**: ``k_idx``
  A 4D tensor of type *T* with shape ``[B, L, H_ik, D_idx]``.
  Current-step projected indexer keys, emitted by the same external ``index_qk_proj``. The index-key head
  count ``H_ik`` is **derived from this shape** (configurable; never hard-coded to ``1``). Appended to the
  pending block and pooled into completed-block summaries. **Required.**

* **2**: ``k_rope_cos_sin``
  A 3D tensor of type *T* with shape ``[B, L, 2 * indexer_rotary_dim]``.
  Per-token RoPE ``(cos, sin)`` for the key side of this step, prepared by the caller from the full table using
  the appropriate (possibly M-RoPE) key coordinates. Used to refresh the pending block-start RoPE phase when a
  new block begins. The ``(cos, sin)`` pairs are laid out in the reference ``rotate_half`` convention. **Required.**

* **3**: ``summary_past``
  A 4D tensor of type *T* with shape ``[B, H_ik, N_c, D_idx]``.
  Mean-pooled + RMSNorm'ed + block-start-RoPE'ed summary keys of the completed blocks so far (``N_c`` dynamic).
  Wired from a ``ReadValue`` node. **Required.**

* **4**: ``indexer_k_past``
  A 4D tensor of type *T* with shape ``[B, H_ik, P, D_idx]``.
  Projected, unrotated keys of the currently incomplete block so far, with ``0 <= P < compress_ratio``. Wired
  from a ``ReadValue`` node. **Required.**

* **5**: ``indexer_k_start_past``
  A 2D tensor of type *T* with shape ``[B, 2 * indexer_rotary_dim]``.
  RoPE ``(cos, sin)`` at the block-start position of the pending block. Wired from a ``ReadValue`` node.
  **Required.**

* **6**: ``k_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``.
  RMSNorm weight for pooled block keys. **Required.**


**Outputs**

* **0**: ``sel_indices``
  A 4D tensor of type *T_IND* with shape ``[B, H_sel, L, K_max]``.
  Selected *block* indices per selection head as a **strictly ascending, duplicate-free** sequence in the
  contiguous prefix ``sel_indices[..., :sel_count]`` (the causal-diagonal / incomplete block, when present, is
  appended immediately after the selected complete blocks and is always greater than them, so the prefix stays
  ascending). ``-1`` padded only past ``sel_count`` (no interior holes). The standard form has no ``B_q``: one
  selection per query token, so the third dimension is ``L``. **Required.**

* **1**: ``sel_count``
  A 3D tensor of type *T_IND* with shape ``[B, H_sel, L]``.
  Number of valid selected blocks per selection head per query. **Required.**

* **2**: ``summary_present``
  A 4D tensor of type *T* with shape ``[B, H_ik, floor((S_0 + L) / r), D_idx]``.
  Updated completed-block summary state, wired to an ``Assign`` node. **Required.**

* **3**: ``indexer_k_present``
  A 4D tensor of type *T* with shape ``[B, H_ik, (S_0 + L) % r, D_idx]``.
  Updated incomplete-block key state, wired to an ``Assign`` node. **Required.**

* **4**: ``indexer_k_start_present``
  A 2D tensor of type *T* with shape ``[B, 2 * indexer_rotary_dim]``.
  Updated block-start RoPE phase, wired to an ``Assign`` node. **Required.**


**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` or ``int64``.
* ``q_idx`` and ``k_idx`` share the first two dims ``[B, L]``; both last dimensions equal ``indexer_head_dim``.
  ``q_idx`` penultimate dimension is ``H_iq`` and ``k_idx`` penultimate dimension is ``H_ik``, with
  ``H_iq % H_ik == 0`` and ``G = H_iq / H_ik`` index-query heads sharing each index-key head.
* ``H_ik`` (index-key heads) is taken from ``k_idx`` shape; ``H_sel`` is ``1`` or ``H_ik``.
* ``summary_past`` dim 1 equals ``H_ik``; ``indexer_k_past`` dim 1 equals ``H_ik``; ``indexer_k_past`` dim 2 satisfies
  ``0 <= P < compress_ratio``.
* ``indexer_k_start_past`` last dimension equals ``2 * indexer_rotary_dim``; ``k_rope_cos_sin`` dim 2 equals the
  same.
* ``sel_indices`` last dimension ``K_max`` equals ``block_topk + 1``; ``sel_count[..., sh] <= K_max``, and the
  valid prefix ``sel_indices[..., sh, :sel_count[..., sh]]`` is **strictly ascending** and contains no ``-1``
  entries.
* ``summary_present`` dim 2 equals ``floor((S_0 + L) / r)``; ``indexer_k_present`` dim 2 equals
  ``(S_0 + L) % r``, where ``S_0 = summary_past.shape[2] * r + indexer_k_past.shape[2]``.
* When the selection for a row is empty (``sel_count == 0``), the consumer must produce a zero output vector for
  that row.


**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Example**

The example below shows a single query token (``B=1``, ``L=1``) at logical position ``S_0 + l = 9`` (``S_0 = 9``,
one token in the step) with ``compress_ratio = 4``, ``block_topk = 2`` (so ``K_max = block_topk + 1 = 3``),
``h_sel = 1``, ``H_iq = 4``, ``H_ik = 1``, ``indexer_head_dim = 128``, ``indexer_rotary_dim = 128``. The history
before this step covers blocks ``{0:[0..3], 1:[4..7]}`` (complete) and block ``2:[8]`` (the causal-diagonal /
incomplete block). Causally-valid complete blocks for ``pos = 9`` are those with ``b < (9+1)//4 = 2``, i.e.
``{0, 1}``. After a stable top-2 by score over ``{0, 1}`` the selection head emits the top-2 complete blocks in
**strictly ascending** order ``{0, 1}`` plus the causal-diagonal block ``2`` written immediately after them
(always greater than the complete blocks, so the prefix stays ascending), yielding
``sel_indices = [[[[0, 1, 2]]]]`` and ``sel_count = [[[3]]]``. With ``H_iq = 4`` and ``H_ik = 1``, all four
index-query heads participate in the single selection head's score.

.. code-block:: xml
   :force:

   <layer ... type="QSAIndexer">
       <data block_topk="2" h_sel="1" indexer_head_dim="128"
             indexer_rotary_dim="128" compress_ratio="4" eps="1e-6"
             scale="0.088388"/>
       <input>
           <port id="0">   <!-- q_idx: [B, L, H_iq, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="1">   <!-- k_idx: [B, L, H_ik, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>1</dim><dim>128</dim>
           </port>
           <port id="2">   <!-- k_rope_cos_sin: [B, L, 2*rotary_dim] -->
               <dim>1</dim><dim>1</dim><dim>256</dim>
           </port>
           <port id="3">   <!-- summary_past: [B, H_ik, N_c, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>2</dim><dim>128</dim>
           </port>
           <port id="4">   <!-- indexer_k_past: [B, H_ik, P, D_idx], 0 <= P < r -->
               <dim>1</dim><dim>1</dim><dim>1</dim><dim>128</dim>
           </port>
           <port id="5">   <!-- indexer_k_start_past: [B, 2*rotary_dim] -->
               <dim>1</dim><dim>256</dim>
           </port>
           <port id="6">   <!-- k_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
       </input>
       <output>
           <port id="7" precision="I32">  <!-- sel_indices: [B, H_sel, L, K_max] -->
               <dim>1</dim><dim>1</dim><dim>1</dim><dim>3</dim>
           </port>
           <port id="8" precision="I32">  <!-- sel_count: [B, H_sel, L] -->
               <dim>1</dim><dim>1</dim><dim>1</dim>
           </port>
           <port id="9">   <!-- summary_present: [B, H_ik, floor((S0+L)/r), D_idx] -->
               <dim>1</dim><dim>1</dim><dim>2</dim><dim>128</dim>
           </port>
           <port id="10">  <!-- indexer_k_present: [B, H_ik, (S0+L)%r, D_idx] -->
               <dim>1</dim><dim>1</dim><dim>2</dim><dim>128</dim>
           </port>
           <port id="11">  <!-- indexer_k_start_present: [B, 2*rotary_dim] -->
               <dim>1</dim><dim>256</dim>
           </port>
       </output>
   </layer>