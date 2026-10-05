.. {#openvino_docs_ops_internal_QSAIndexer}

QSAIndexer
==========

.. meta::
  :description: Learn about QSAIndexer - a model-specific, weight-free block-level indexer for Qwen Sparse Attention that consumes an externally projected query and token keys, maintains an incremental pooled-key summary cache, and emits a clean (sel_indices, sel_count) block selection contract for the unified SparseSDPA / SparsePA consumers.

**Versioned name**: *QSAIndexer*

**Category**: *Internal*

**Short description**:
The *QSAIndexer* is the **model-specific plan producer** for Qwen Sparse Attention (QSA). It consumes an
already-projected indexer query ``q_idx`` and per-token projected keys (produced by an ordinary
``FullyConnected`` / ``MatMul`` outside the op), then performs the *K-side compression chain* — per-token
or pooled key compression (mean-pool), GemmaRMSNorm, and RoPE at the historical block start — maintains the
incremental ``summary_cache``, scores each block with a per-head ReLU dot-product reduced across heads, and
selects the top-:math:`K` blocks. It emits a **clean, producer-complete** ``(sel_indices, sel_count)`` block
selection contract. The unified consumers ``SparseSDPA`` / ``SparsePA`` are deliberately *plan consumers*:
they only gather KV at the selected blocks and perform online-softmax attention. No ``index_weights``, no
output gate, no RoPE/norm parameters, and no ``r-1`` tail hack leak from the producer into the consumer.

**Detailed description**

The *QSAIndexer* realizes the *producer* side of the sparse-attention split. It is *model-specific*: it
encodes the exact QSA reference math (mean-pool :math:`\to` RMSNorm :math:`\to` block-start RoPE :math:`\to`
per-head ReLU score :math:`\to` top-k), in contrast to the *unified* ``SparseSDPA`` / ``SparsePA`` consumers
which are model-agnostic. The indexer GEMM weight (``index_qk_proj``) is **kept outside** this op as a
standard ``FullyConnected`` / ``MatMul`` for four reasons:

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

**Data flow and cache organization**

The K-side history is maintained as two caches separate from the main KV cache:

- ``raw_k_cache`` ``[num_blocks, 1, block_size, D_idx]``: the projected, per-token indexer keys written by the
  external GEMM and appended in place. RoPE is *not* baked here.
- ``summary_cache`` ``[num_blocks, 1, block_size / r, D_idx]``: the mean-pooled, RMSNorm'ed, block-start-RoPE'ed
  summary key of each *completed* block. This is the incrementally-updated artifact that makes decode
  :math:`O(1)` per new token.

A block is *completed* once exactly :math:`r` raw keys have been appended. The last, partially-filled block of
a sequence is *not* summarized into ``summary_cache``; instead it is emitted as a **valid block index** in the
output (the producer-completeness rule), so the consumer gathers it as a partial block without any ``r-1`` tail
hack in the selection encoding.

**Completeness of the emitted selection**

Every block a query must attend to is emitted as a *valid* block index by the producer, including:

- the **sink** block(s),
- the **sliding-window** blocks (when applicable),
- the **top-k scored** complete blocks,
- the **causal diagonal / incomplete block** (emitted as the last valid block index of the selection head, so
  the consumer knows to gather it partially).

This is the *producer-completeness* contract: ``SparseSDPA`` / ``SparsePA`` never re-derive what to attend to;
they gather exactly the blocks named by ``sel_indices`` and counted by ``sel_count``.

**Score computation**

For query position :math:`p` with projected query :math:`q \in \mathbb{R}^{H_{sel}\times D_{idx}}` and block
:math:`b` whose summary key is :math:`\bar{k}_b \in \mathbb{R}^{D_{idx}}`:

.. math::

    s_b = \frac{1}{\sqrt{D_{idx}}} \sum_{h=1}^{H_{sel}} \mathrm{ReLU}\!\left( \langle \mathrm{RoPE}_{pos(p)}(\mathrm{RMSNorm}(q_h)),\, \bar{k}_{b,h} \rangle \right)

where :math:`\bar{k}_b = \mathrm{RoPE}_{b\cdot r}(\mathrm{RMSNorm}(\tfrac{1}{r}\sum_{j=0}^{r-1} k_{b\cdot r + j}))`.
The ReLU is applied *per head before reduction over heads*, matching the reference QSA implementation. The
indexer query :math:`q` arrives already projected (external GEMM); the op applies the query RMSNorm and partial
RoPE from ``q_norm_weight`` and ``key_cos_sin`` at ``position_ids[p]``.

**Selection, tie-breaking, and top-k**

- The number of selected *complete* blocks per selection head is
  :math:`K_{max} = \lfloor \mathrm{budget} / \mathrm{compress\_ratio} \rfloor`. If fewer complete blocks exist
  (prefill, short context), all complete blocks are selected and ``sel_count`` records the smaller number.
- **Tie-breaking**: scores are reduced by head and normalized once. Ties in ``topk`` are broken by *smaller
  block index* (earlier blocks win), matching the reference PyTorch ``topk`` with default ``sorted=False`` under
  the fused-kernel contract. Stable across CPU and GPU so greedy decode is reproducible.
- **Selection heads**: :math:`H_{sel} \in \{1, H_{kv}\}`. QSA uses a single shared selection (:math:`H_{sel}=1`);
  models with per-KV-head selection set :math:`H_{sel}=H_{kv}`. Per-query-head selection for GQA is rejected.

**Block grid and causal rules**

The KV history of a sequence is partitioned into blocks of :math:`r` tokens. Block :math:`b` covers tokens
:math:`[b \cdot r, (b+1) \cdot r - 1]`. Scoring is causal: a query at position :math:`p` may only select blocks
with :math:`(b{+}1) \cdot r - 1 \le p`, plus the current incomplete block (the causal diagonal). When the
``is_causal`` attribute is true the emitted ``sel_indices`` are already causal-complete; otherwise the full
top-k is emitted and causality is imposed downstream.

**Pseudo-code (numpy)**

.. code-block:: py
    :force:

    def qsa_indexer(q_idx, raw_k_cache, summary_cache, q_norm_weight, k_norm_weight,
                    key_cos_sin, position_ids, subsequence_begins,
                    *, compress_ratio, budget, block_topk, h_sel, indexer_head_dim,
                    indexer_rotary_dim, num_sequences, eps, scale, is_causal):
        # q_idx:            [tokens, H_sel * D_idx]   (projected by external GEMM)
        # raw_k_cache:      [num_blocks, 1, block_size, D_idx]  (block_size == compress_ratio)
        # summary_cache:    [num_blocks, 1, block_size//r, D_idx]
        # returns: sel_indices [tokens, H_sel, K_max]   (block indices, int32; -1 = pad)
        #          sel_count   [tokens, H_sel]          (valid block count, int32)
        tokens = q_idx.shape[0]
        D = q_idx.shape[1]
        Hsel, Di = h_sel, indexer_head_dim

        # 1. Query RMSNorm + partial RoPE at the query's own position.
        q = reshape(q_idx, (tokens, Hsel, Di))
        q = rms_norm(q, q_norm_weight, eps)                              # [tokens, Hsel, Di]
        q = rope(q[..., :indexer_rotary_dim], key_cos_sin[position_ids]) # [tokens, Hsel, Di]

        # 2. Append raw projected keys; pool completed blocks incrementally into summary_cache.
        #    (Only touches the current incomplete block per step: O(r*Di).)
        for s in range(num_sequences):
            start, end = subsequence_begins[s], subsequence_begins[s + 1]
            for t in range(start, end):
                # raw projected key k[t] appended in place to raw_k_cache; when the block completes,
                #   bar_k = RMSNorm(mean of r raw keys) then RoPE at block-start position
                maybe_complete_block(k[t], raw_k_cache, summary_cache, t, k_norm_weight,
                                     key_cos_sin, compress_ratio, eps)

        # 3. Score every complete block against each selection head (causal filter), reduce over heads.
        num_complete = number_of_complete_blocks()                        # from summary_cache
        scores = zeros((tokens, Hsel, num_complete))
        for p in range(tokens):
            pos = position_ids[p]
            for sh in range(Hsel):
                for b in range(num_complete):
                    if is_causal and (b + 1) * compress_ratio - 1 > pos:
                        scores[p, sh, b] = -inf
                        continue
                    bar = summary_cache[b, 0]                             # [Di]
                    dot = sum(q[p, sh] * bar, axis=-1)                    # scalar
                    scores[p, sh, b] = relu(dot) * scale

        # 4. Top-k block selection per selection head; ties broken by smaller block index.
        sel_indices = full((tokens, Hsel, block_topk), -1, dtype=int32)
        sel_count   = zeros((tokens, Hsel), dtype=int32)
        for p in range(tokens):
            for sh in range(Hsel):
                order = argsort(-scores[p, sh], kind='stable')            # stable => earlier block wins
                k = min(block_topk, num_complete)
                picked = order[:k]
                sel_indices[p, sh, :k] = picked
                sel_count[p, sh] = k
                # Producer-completeness: if the query is inside an incomplete block, append that block
                #   as the last valid index so the consumer gathers the causal diagonal.
                inc = incomplete_block_index(p, position_ids[p], compress_ratio)
                if is_causal and inc is not None and k < block_topk:
                    sel_indices[p, sh, k] = inc
                    sel_count[p, sh] = k + 1
        return sel_indices, sel_count


**Attributes**

* *compress_ratio*

  * **Description**: :math:`r`, the number of raw token keys pooled into one block key.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``compress_ratio >= 1``. ``block_size == compress_ratio``.

* *budget*

  * **Description**: :math:`B`, the maximum number of selected tokens per query head across complete blocks.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``budget % compress_ratio == 0``.

* *block_topk*

  * **Description**: Maximum number of complete blocks selected per selection head,
    ``== budget // compress_ratio``. Also the ``K_max`` of the ``sel_indices`` output.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``block_topk >= 1``.

* *h_sel* (``indexer_heads``)

  * **Description**: Number of selection heads :math:`H_{sel}`. Must be ``1`` or ``num_kv_heads`` of the
    main attention.
  * **Type**: ``int``
  * **Required**: *yes*
  * **Constraints**: ``h_sel >= 1``; ``h_sel == 1`` (QSA default) or ``h_sel == main num_kv_heads``.

* *indexer_head_dim*

  * **Description**: Head dimension :math:`D_{idx}` of the indexer.
  * **Type**: ``int``
  * **Required**: *yes*

* *indexer_rotary_dim*

  * **Description**: RoPE dimension used by the indexer (partial rotary). Must satisfy
    ``indexer_rotary_dim <= indexer_head_dim``.
  * **Type**: ``int``
  * **Required**: *yes*

* *k_norm_type*

  * **Description**: Normalization applied to the pooled block key. Must be ``rms`` (RMSNorm / GemmaRMSNorm).
  * **Type**: ``enum``
  * **Required**: *yes*
  * **Supported values**: ``rms``

* *eps*

  * **Description**: Epsilon for RMSNorm.
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*

* *scale*

  * **Description**: Score normalization constant, ``1.0 / sqrt(indexer_head_dim)``.
  * **Type**: ``float``
  * **Required**: *yes*

* *is_causal*

  * **Description**: Whether scoring is causal-complete inside the op (emits the causal diagonal block).
  * **Type**: ``bool``
  * **Default value**: ``true``
  * **Required**: *no*


**Inputs**

* **0**: ``q_idx``
  A 2D tensor of type *T* with shape ``[tokens, H_sel * D_idx]``.
  Projected indexer queries emitted by the external ``FullyConnected`` / ``MatMul`` (``index_qk_proj``).
  **Required.**

* **1**: ``raw_k_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, 1, block_size, D_idx]``.
  Projected per-token indexer keys, written by the external GEMM and appended in place. **Required.**

* **2**: ``summary_cache``
  A 4D tensor of type *T* with shape ``[num_blocks, 1, block_size // compress_ratio, D_idx]``.
  Mean-pooled + RMSNorm'ed + block-start-RoPE'ed summary key of each completed block. Updated in place.
  **Required.**

* **3**: ``q_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``.
  RMSNorm weight for indexer queries. **Required.**

* **4**: ``k_norm_weight``
  A 1D tensor of type *T* with shape ``[D_idx]``.
  RMSNorm weight for pooled block keys. **Required.**

* **5**: ``key_cos_sin``
  A 2D tensor of type *T* with shape ``[max_positions, 2 * indexer_rotary_dim]``.
  Precomputed RoPE cosine/sine table, ``(cos, sin)`` interleaved per rotary dimension. Required so that
  M-RoPE and variable-length batching remain expressible without materializing per-position grids. **Required.**

* **6**: ``position_ids``
  A 1D tensor of type *T_IND* with shape ``[tokens]``.
  Absolute token positions (or M-RoPE rank when the model uses multimodal rotary embeddings). **Required.**

* **7**: ``subsequence_begins``
  A 1D tensor of type *T_IND* with shape ``[num_sequences + 1]``.
  Splits ``tokens`` among sequences. Sequence ``s`` uses tokens
  ``[subsequence_begins[s] : subsequence_begins[s + 1]]``. **Required.**


**Outputs**

* **0**: ``sel_indices``
  A 3D tensor of type *T_IND* with shape ``[tokens, H_sel, block_topk]``.
  Selected *block* indices per selection head, ``-1`` padded past ``sel_count``. The last valid index of a
  selection head may be the causal-diagonal / incomplete block (producer-completeness). **Required.**

* **1**: ``sel_count``
  A 2D tensor of type *T_IND* with shape ``[tokens, H_sel]``.
  Number of valid selected blocks per selection head per query. **Required.**


**Shape inference and type rules**

* ``T`` is a floating-point type (``float32``, ``float16``, ``bfloat16``).
* ``T_IND`` is ``int32`` or ``int64``.
* ``tokens``, ``num_sequences`` and ``subsequence_begins`` must be consistent:
  ``subsequence_begins[0] == 0`` and ``subsequence_begins[-1] == tokens``.
* ``q_idx`` last dimension equals ``h_sel * indexer_head_dim``.
* ``num_blocks`` and ``block_size`` must be consistent with the paged geometry of the calling context; the
  operation itself does not derive them from ``q_idx``.
* ``sel_indices`` last dimension is fixed at ``block_topk``; ``sel_count[t, sh] <= block_topk``.
* When ``is_causal`` is false, ``sel_indices`` may reference any block and downstream must enforce causality.


**Types**

* *T*: any floating point type.
* *T_IND*: ``int32`` or ``int64``.


**Example**

The example below shows a single query token against a history of 9 projected raw tokens with
``compress_ratio = 4``, ``budget = 8``, ``block_topk = 2``, ``h_sel = 1``. Blocks ``{0:[0..3], 1:[4..7]}`` are
complete; block ``2:[8]`` is the causal diagonal / incomplete block. After a stable top-2 by score the selection
head emits blocks ``{1, 0}`` plus the causal-diagonal block ``2`` (producer-completeness), yielding
``sel_indices = [[[1, 0, 2]]]`` and ``sel_count = [[3]]``.

.. code-block:: xml
   :force:

   <layer ... type="QSAIndexer">
       <data compress_ratio="4" budget="8" block_topk="2"
             h_sel="1" indexer_head_dim="128"
             indexer_rotary_dim="128" k_norm_type="rms" eps="1e-6"
             scale="0.088388" is_causal="true"/>
       <input>
           <port id="0">   <!-- q_idx: [tokens, H_sel*D_idx] -->
               <dim>2</dim><dim>128</dim>
           </port>
           <port id="1">   <!-- raw_k_cache: [num_blocks, 1, block_size, D_idx] -->
               <dim>4</dim><dim>1</dim><dim>4</dim><dim>128</dim>
           </port>
           <port id="2">   <!-- summary_cache: [num_blocks, 1, block_size//r, D_idx] -->
               <dim>4</dim><dim>1</dim><dim>1</dim><dim>128</dim>
           </port>
           <port id="3">   <!-- q_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
           <port id="4">   <!-- k_norm_weight: [D_idx] -->
               <dim>128</dim>
           </port>
           <port id="5">   <!-- key_cos_sin: [max_positions, 2*rotary_dim] -->
               <dim>262144</dim><dim>256</dim>
           </port>
           <port id="6">   <!-- position_ids: [tokens] -->
               <dim>2</dim>
           </port>
           <port id="7">   <!-- subsequence_begins: [num_sequences+1] -->
               <dim>2</dim>
           </port>
       </input>
       <output>
           <port id="0" precision="I32">   <!-- sel_indices: [tokens, H_sel, block_topk] -->
               <dim>2</dim><dim>1</dim><dim>2</dim>
           </port>
           <port id="1" precision="I32">   <!-- sel_count: [tokens, H_sel] -->
               <dim>2</dim><dim>1</dim>
           </port>
       </output>
   </layer>