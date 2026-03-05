PagedAttentionExtension
=======================

.. meta::
   :description: Learn about PagedAttentionExtension - a stateful paged KV-cache multi-head
                 attention operator with GQA, RoPE re-rotation, score aggregation, and
                 adaptive KV eviction support.

**Versioned name**: *PagedAttentionExtension-13*

**Category**: *Sequence processing*

**Short description**: *PagedAttentionExtension* implements causal multi-head attention
with Grouped Query Attention (GQA) over a *block-paged* key-value cache.  Unlike standard
attention, the KV cache is not a contiguous padded tensor - it is split into equal-size
*blocks* that may be scattered across physical memory and are referenced through a *block
table*.  This removes per-sequence padding overhead and enables blocks to be reused or
evicted independently.  The operator is *stateful*: KV data written in one inference call
(prefill) persists and is reused in subsequent calls (decode steps) without re-computation.


.. rubric:: Concepts

The following terms are used throughout this document.

**Token** - a single row in the flattened Q/K/V matrices, corresponding to one input
element of one sequence.

**Sequence** - a logical prompt or conversation turn.  A batch may contain multiple
sequences whose tokens are packed contiguously in the input tensors.

**Block** - a fixed-size chunk of ``block_size`` consecutive token positions inside the
KV cache.  Every sequence's cache is covered by one or more blocks; the mapping is given
by the block table.

**Block table** - the pair of ``block_indices`` and ``block_indices_begins`` inputs.
``block_indices_begins[s]`` to ``block_indices_begins[s+1]`` indexes into
``block_indices``; those values are the physical block IDs for sequence ``s``, in order.
A value of ``-1`` is a padding sentinel (no valid data at that slot).

**Past length** - ``past_lens[s]``: the number of tokens already cached for sequence
``s`` from an earlier inference call.

**New tokens** - the tokens being processed in the current call.  For a prefill step,
this is many tokens per sequence; for a decode step, typically one per sequence.

**GQA (Grouped Query Attention)** - multiple query heads share one KV head:
``q_heads = group * kv_heads``.  All query heads in the same group read from the same
cached K and V vectors.

**Score aggregation** - output 1: a per-KV-position accumulation of attention weights
summed over all query heads and over the query tokens inside the ``score_aggregation_window``.
Used downstream for KV-cache importance estimation.

**ALiBi** - Attention with Linear Biases.  A linear penalty ``slope[h] * (key_pos - query_pos)``
is added to each logit before softmax, reducing attention to distant keys.

**RoPE re-rotation** - when tokens are evicted from the front of a sequence and the
remaining tokens shift to new logical positions, the cached keys carry stale rotary
position encodings.  Re-rotation corrects them on the fly using a precomputed trig LUT.

**Xattention** - sparse attention where only the most important
``(query_block, key_block)`` pairs are computed at full precision.  The importance of
each block pair is estimated cheaply with strided dot products, then thresholded.

**Sinks** - a per-head scalar added as a virtual attention target during softmax.  It
participates in the normalization denominator (preventing weights from summing to 1) but
produces no contribution to the output.  This attenuates attention scores across all
real key positions uniformly.

**Adaptive RKV eviction** - a method for deciding which KV-cache blocks to evict by
computing pairwise key-vector diversity within the eviction zone.  Output 2 carries the
resulting diversity scores so the caller can rank blocks for eviction.


**Detailed description**:

*PagedAttentionExtension* operates in three phases: (1) per-call cache initialization and
state refresh, (2) attention computation with optional auxiliary features, and (3) optional
score and diversity output production.  The following pseudocode describes the full algorithm.
``cache_manager`` is an internal object attached to the op node; it manages physical KV
storage across multiple inference calls.

.. code-block:: py

   def PagedAttentionExtension(
       query,                                   # input  0  [T, Hq*S]
       key,                                     # input  1  [T, Hkv*S]
       value,                                   # input  2  [T, Hkv*Sv]
       key_cache,                               # input  3  [B, Hkv, Bs, S]
       value_cache,                             # input  4  [B, Hkv, Bs, Sv]
       past_lens,                               # input  5  [B_seq]        i32
       subsequence_begins,                      # input  6  [B_seq+1]      i32
       block_indices,                           # input  7  [Nb]           i32
       block_indices_begins,                    # input  8  [B_seq+1]      i32
       scale,                                   # input  9  []             real
       sliding_window,                          # input 10  []             i32
       alibi_slopes,                            # input 11  [Hq] or empty  real
       max_context_len,                         # input 12  []             i32
       score_aggregation_window,                # input 13  [] or [B_seq]  i32
       rotated_block_indices,                   # input 14  [Nrot] or empty  i32
       rotation_deltas,                         # input 15  [Nrot] or [Nrot,Bs] or empty  i32
       rotation_trig_lut,                       # input 16  [C,S] or empty  f16/f32
       xattention_threshold,                    # input 17  [] or [B_seq] or empty  f16/f32
       xattention_block_size,                   # input 18  []             i32
       xattention_stride,                       # input 19  []             i32
       sinks,                                   # input 20  [1,Hq,1,1] or empty  real
       adaptive_rkv_start_size,                 # input 21  []             i32
       adaptive_rkv_evictable_sizes,            # input 22  [B_seq] or empty  i32
       adaptive_rkv_diversity_block_set_indices,         # input 23  [Narkv] or empty  i32
       adaptive_rkv_diversity_block_set_indices_begins,  # input 24  [B_seq+1] or empty  i32
   ):
       # ---------------------------------------------------------------
       # Phase 0 - cache initialization (runs once per op node lifetime)
       # ---------------------------------------------------------------
       # On the very first call, copies key_cache, value_cache, and the
       # block table into the internal cache manager, and builds one
       # SequenceState per sequence using past_lens.  All subsequent calls
       # skip this step and find the data already present.
       if not cache_manager.is_initialized():
           cache_manager.init(key_cache, value_cache,
                              block_indices, block_indices_begins, past_lens)

       # Refresh each sequence's logical length to match this step's past_lens.
       cache_manager.begin_step(past_lens)

       # ---------------------------------------------------------------
       # Phase 1 - scalar parameter extraction
       # ---------------------------------------------------------------
       scale_f    = float(scale)
       sw         = int(sliding_window)    # 0 = full causal attention
       max_ctx    = int(max_context_len)   # 0 = unlimited
       score_win  = score_aggregation_window  # scalar or [B_seq]; 0 = disabled (output 1 zeroed)
       group      = Hq // Hkv             # GQA group size

       # ---------------------------------------------------------------
       # Phase 2 - optional xattention sparse mask (prefill, B_seq == 1)
       # ---------------------------------------------------------------
       # Activated only when: B_seq == 1, past_lens[0] == 0, new_len > 1,
       # xattention_block_size > 0, xattention_stride > 0.
       #
       # Step 2a: divide [0, new_len) into query blocks of size xattn_block_size
       #          and key blocks of the same size.
       # Step 2b (strided Q·K estimation):
       #   For each offset off in [0, xattn_stride):
       #     For each strided query position qi and key position ki (causal):
       #       Compute dot(query[qi * xattn_stride + off],
       #                   key [ki * xattn_stride + off]) * (scale_f / xattn_stride)
       #       Accumulate into attn_strided[qi, ki].
       #   Apply causal masking, then row-wise softmax over attn_strided.
       # Step 2c (block aggregation):
       #   block_sums[qb, kb] = sum of attn_strided[qi, ki]
       #       for qi in [qb*num_per_block, (qb+1)*num_per_block)
       #       and ki in [kb*num_per_block, (kb+1)*num_per_block)
       # Step 2d (threshold selection per query block):
       #   For each query block qb:
       #     total = sum(block_sums[qb, 0..qb])       # causal budget
       #     target = total * threshold
       #     Always keep block 0 (first) and block qb (diagonal).
       #     Sort remaining blocks by importance descending.
       #     Greedily add blocks until cumulative sum >= target.
       #     xattn_mask[h][qb][kb] = True iff block (qb,kb) was selected.
       #   kb > qb is always masked (strict causality).
       xattn_mask = build_xattn_mask(...)  # described above; None if inactive

       # ---------------------------------------------------------------
       # Phase 3 - per-sequence, per-new-token attention loop
       # ---------------------------------------------------------------
       scores_acc = zeros(T + sum(past_lens))   # flat; only meaningful if output 1 requested

       for s in range(B_seq):
           past    = past_lens[s]
           t_begin = subsequence_begins[s]
           t_end   = subsequence_begins[s + 1]
           new_len = t_end - t_begin

           # Window of query tokens that contribute to the score output.
           # score_win == 0 → disabled: output 1 is zero-filled for this sequence.
           # score_win  > 0 → only the last score_win tokens of this sequence contribute.
           win = score_win if isinstance(score_win, int) else score_win[s]

           for i in range(new_len):
               token = t_begin + i
               qpos  = past + i   # logical causal position in the full sequence

               # Write the new key and value vectors into the paged cache.
               # The cache manager allocates a new block if needed.
               cache_manager.write_kv(s, qpos, key[token], value[token])

               # Determine the causal attention window [start, qpos].
               start = 0
               if max_ctx > 0:
                   start = max(start, qpos + 1 - max_ctx)
               if sw > 0:
                   start = max(start, qpos + 1 - sw)

               # Whether this query token's weights are aggregated into output 1.
               # win == 0: disabled - no accumulation, output 1 stays zero.
               # win  > 0: only the last win tokens of this sequence contribute.
               # win  < 0: all new tokens contribute (unbounded window).
               in_score_window = (win != 0) and ((win < 0) or ((new_len - i) <= win))

               for h in range(Hq):
                   kvh   = h // group
                   qvec  = query[token, h * S : (h+1) * S]
                   logits = []

                   for kpos in range(start, qpos + 1):
                       k_vec = cache_manager.get_key(s, kpos, kvh)
                       if k_vec is None:   # evicted token - treat as -inf
                           logits.append(-inf)
                           continue

                       # RoPE re-rotation: correct stale position encodings in
                       # pre-existing cached keys that belong to a rotated block.
                       if (kpos < past
                               and rotated_block_indices is not None
                               and block_of(kpos) in rotated_block_indices):
                           trig_row = get_rotation_row(kpos, rotated_block_indices,
                                                       rotation_deltas)
                           k_vec = rope_rotate(k_vec, rotation_trig_lut, trig_row)
                           # rope_rotate applies split-half LLaMA rotation:
                           #   half = S // 2
                           #   x'[d]      = x[d]*cos[d]      - x[d+half]*sin[d]
                           #   x'[d+half] = x[d]*sin[d]      + x[d+half]*cos[d]
                           #   where cos[d], sin[d] = trig_lut[trig_row, d], trig_lut[trig_row, d+half]

                       dot = sum(qvec[d] * k_vec[d] for d in range(S))
                       logit = dot * scale_f

                       # ALiBi: linear penalty for distance (always <= 0)
                       if alibi_slopes is not None:
                           logit += alibi_slopes[h] * (kpos - qpos)

                       # Xattention: block-level sparsity from Phase 2
                       if xattn_mask is not None:
                           qb = qpos // xattention_block_size
                           kb = kpos // xattention_block_size
                           if not xattn_mask[h][qb][kb]:
                               logit = -inf

                       logits.append(logit)

                   # Softmax with optional sink.
                   # The sink is a scalar per head; it augments the softmax
                   # denominator without producing an output component, so
                   # all real attention weights are attenuated uniformly.
                   sink_val = sinks[0, h, 0, 0] if sinks is not None else None
                   weights  = softmax_with_optional_sink(logits, sink_val)

                   # Weighted sum over value vectors → output 0
                   for t, kpos in enumerate(range(start, qpos + 1)):
                       v_vec = cache_manager.get_value(s, kpos, kvh)
                       output[token, h * Sv : (h+1) * Sv] += weights[t] * v_vec

                   # Score accumulation → output 1
                   if in_score_window:
                       offset = score_offset(s)   # sum of (past+new) for earlier sequences
                       for t, kpos in enumerate(range(start, qpos + 1)):
                           scores_acc[offset + kpos] += weights[t]

       # ---------------------------------------------------------------
       # Phase 4 - outputs
       # ---------------------------------------------------------------
       out_attention = output          # shape [T, Hq*Sv]

       # Output 1: flat score buffer, one region per sequence.
       # Region for sequence s spans [offset_s, offset_s + past_s + new_len_s).
       # Value at position kpos = sum over qualifying query tokens and all Hq
       # heads of the softmax weight pointing from that token to kpos.
       out_scores = scores_acc         # shape [T + sum(past_lens)]

       # Output 2: diversity scores for adaptive KV eviction.
       # Assembled only when adaptive_rkv_evictable_sizes is non-empty.
       out_diversity = []
       if adaptive_rkv_evictable_sizes is not None:
           Bs = key_cache.shape[2]
           for s in range(B_seq):
               evict_size  = adaptive_rkv_evictable_sizes[s]
               total_toks  = past_lens[s] + new_len_s
               start_size  = int(adaptive_rkv_start_size)
               if total_toks < start_size + evict_size:
                   continue
               # Collect all key vectors for this sequence from the cache.
               key_data = [cache_manager.get_key(s, kpos, kvh)
                           for kvh in range(Hkv)
                           for kpos in range(total_toks)]
               # Compute pairwise block-level diversity matrix.
               # Result shape: [evict_size // Bs, evict_size]
               diversity = compute_block_diversity(key_data, evict_size, Bs)
               out_diversity.extend(diversity.flat)
               # Total output 2 length = sum_s (evict_size[s]^2 / Bs)

       return out_attention, out_scores, out_diversity


**Attributes**

This operation has no serialized graph attributes.  The following per-node API controls
are used at the time the model is built or compiled.

* *output type overrides* (``set_out_type`` / ``get_out_type``)

  * **Description**: Override the element type of output ``i`` (``i`` ∈ {0, 1, 2}).
    When not overridden, all outputs inherit the element type of input 0 (query).  Plugin
    implementations may force individual output types; for example the CPU plugin forces
    output 1 to ``f32`` regardless of the inference precision.
  * **Type**: ``ov::element::Type``
  * **Required**: *no*

* *cache manager* (``set_cache_manager``)

  * **Description**: Attach an internal ``PagedCacheManager`` instance to the op node.
    This object holds all physical KV-cache state across inference calls.  It must be
    attached before compilation.  The ``AttachCacheManagerToPagedAttention`` transformation
    pass creates and attaches a manager automatically when included in the plugin's
    transformation pipeline.  Models constructed programmatically (e.g. in tests) may call ``set_cache_manager`` directly.
  * **Required**: *yes* (must be set before ``compile_model``)


**Inputs**

* **1**: ``query`` - 2D tensor of type *T*, shape ``[T, Hq*S]``.  Rows are query tokens
  packed across all sequences; ``Hq`` query heads of size ``S`` are concatenated in
  the feature dimension.  **Required.**

* **2**: ``key`` - 2D tensor of type *T*, shape ``[T, Hkv*S]``.  Same token packing as
  ``query``; ``Hkv`` KV heads (``Hkv ≤ Hq``, must divide ``Hq``).  **Required.**

* **3**: ``value`` - 2D tensor of type *T*, shape ``[T, Hkv*Sv]``.  Value head size
  ``Sv`` may differ from the key head size ``S``.  **Required.**

* **4**: ``key_cache`` - tensor of type *T_cache* and shape ``[B, Hkv, Bs, S]`` (rank 4
  canonical; ranks 2–5 accepted during shape inference).  Physical block storage for
  previously cached key vectors.  Used to initialize the internal cache manager on the
  first inference call; its layout determines ``Bs``, ``Hkv``, and ``S``.  **Required.**

* **5**: ``value_cache`` - tensor of type *T_cache* and shape ``[B, Hkv, Bs, Sv]``.
  Physical block storage for previously cached value vectors.  Same first-call semantics.
  **Required.**

* **6**: ``past_lens`` - 1D tensor of type ``i32``, shape ``[B_seq]``.  ``past_lens[s]``
  is the number of tokens already present in the KV cache for sequence ``s`` before this
  inference call.  **Required.**

* **7**: ``subsequence_begins`` - 1D tensor of type ``i32``, shape ``[B_seq+1]``.
  Exclusive scan over the per-sequence new-token counts.  Rows
  ``subsequence_begins[s]`` to ``subsequence_begins[s+1]`` (exclusive) in ``query``,
  ``key``, and ``value`` belong to sequence ``s``.  **Required.**

* **8**: ``block_indices`` - 1D tensor of type ``i32``, shape ``[Nb]``.  Flat list of
  physical block IDs forming the block table.  Sequence ``s`` uses the entries at
  positions ``block_indices_begins[s]`` through ``block_indices_begins[s+1]-1``.
  A value of ``-1`` is a padding sentinel meaning the slot is unused.  **Required.**

* **9**: ``block_indices_begins`` - 1D tensor of type ``i32``, shape ``[B_seq+1]``.
  Per-sequence offsets into ``block_indices``.  **Required.**

* **10**: ``scale`` - scalar tensor of type *T_real*, shape ``[]``.  Attention scale
  factor, typically ``1/sqrt(S)``.  **Required.**

* **11**: ``sliding_window`` - scalar tensor of type ``i32``, shape ``[]``.  Maximum
  number of past token positions each new token can attend to.  ``0`` = unlimited causal
  attention.  When positive, positions further than ``sliding_window`` steps in the past
  are excluded (attention weight clamped to zero).  **Required.**

* **12**: ``alibi_slopes`` - 1D tensor of type *T_real*, shape ``[Hq]``, or an empty
  tensor (``size = 0``).  When provided, adds the ALiBi linear bias
  ``slope[h] * (key_pos - query_pos)`` to every attention logit before softmax, penalizing
  distant keys with a linearly increasing negative offset.  Mutually exclusive with the sliding
  window in the CPU implementation (see implementation notes).  **Optional.**

* **13**: ``max_context_len`` - scalar tensor of type ``i32``, shape ``[]``.  Hard upper
  bound on the number of attended positions, counting from the current query position backwards.
  ``0`` = no upper bound.  Applied in addition to ``sliding_window``; the effective start of
  the attention window is the maximum of both constraints.  **Required.**

* **14**: ``score_aggregation_window`` - scalar or 1D tensor of type ``i32``, shape ``[]``
  or ``[B_seq]``.  Controls which new query tokens contribute to the score aggregation output.
  A value of ``0`` disables score aggregation for that sequence (output 1 is zero-filled).
  A positive value ``N`` means only the last ``N`` new tokens of each sequence contribute.
  A negative value causes all new tokens to contribute (unbounded window).
  Can be a single scalar applied to all sequences, or one value per sequence.  **Required.**

* **15**: ``rotated_block_indices`` - 1D tensor of type ``i32``, shape ``[Nrot]``, or empty.
  Physical block IDs of KV-cache blocks whose keys carry pre-rotated position encodings
  that must be corrected before use.  Empty = RoPE re-rotation is not needed.  **Optional.**

* **16**: ``rotation_deltas`` - tensor of type ``i32``, shape ``[Nrot]`` (1D) or
  ``[Nrot, Bs]`` (2D), or empty.  For each block in ``rotated_block_indices``, provides
  the row index into ``rotation_trig_lut`` to use for re-rotation.  1D form provides one
  row per block; 2D form provides one row per token within the block (finer granularity).
  **Optional.**

* **17**: ``rotation_trig_lut`` - tensor of type ``f16`` or ``f32``, shape ``[C, S]`` or
  ``[C*S]``, or empty.  Look-up table for RoPE position correction.  Row ``r`` contains
  ``[cos_0, ..., cos_{S/2-1}, sin_0, ..., sin_{S/2-1}]`` where ``S/2`` is the half-head
  size.  ``C`` is the number of available rows.  Empty = no RoPE re-rotation.  **Optional.**

* **18**: ``xattention_threshold`` - scalar or 1D tensor of type ``f16`` or ``f32``, shape
  ``[]`` or ``[B_seq]``, or empty.  Attention sparsity threshold for xattention.  For each
  query block, key blocks whose cumulative importance mass covers at least ``threshold``
  fraction of the total causal budget are kept; the rest are masked to ``-inf``.  Empty =
  dense (full) attention.  **Optional.**

* **19**: ``xattention_block_size`` - scalar tensor of type ``i32``, shape ``[]``.  Token
  granularity for xattention block grouping.  Has no effect when xattention is inactive.
  **Required.**

* **20**: ``xattention_stride`` - scalar tensor of type ``i32``, shape ``[]``.  Stride used
  in the strided Q·K importance estimation pass of xattention.  **Required.**

* **21**: ``sinks`` - tensor of type *T*, shape ``[1, Hq, 1, 1]``, or empty.  Per-query-head
  sink value.  Participates in the softmax normalization as a virtual ``exp(sink[h])`` term
  in the denominator, reducing all real attention weights proportionally.  Does not produce
  a component in the output.  Empty = disabled.  **Optional.**

* **22**: ``adaptive_rkv_start_size`` - scalar tensor of type ``i32``, shape ``[]``.
  Number of initial tokens in each sequence that are exempt from eviction scoring (always
  considered maximally important).  Used only when ``adaptive_rkv_evictable_sizes`` is
  non-empty.  **Required.**

* **23**: ``adaptive_rkv_evictable_sizes`` - 1D tensor of type ``i32``, shape ``[B_seq]``,
  or empty.  Per-sequence count of tokens eligible for eviction scoring.  When non-empty,
  activates computation of output 2 (diversity scores).  **Optional.**

* **24**: ``adaptive_rkv_diversity_block_set_indices`` - 1D tensor of type ``i32``, shape
  ``[Narkv]``, or empty.  Flat list of physical block IDs forming the candidate eviction set
  for each sequence.  Passed to plugin kernels for scheduling purposes; the reference
  implementation derives candidates directly from cached key data and does not use this
  input.  **Optional.**

* **25**: ``adaptive_rkv_diversity_block_set_indices_begins`` - 1D tensor of type ``i32``,
  shape ``[B_seq+1]``, or empty.  Per-sequence offsets into input 24.  **Optional.**


**Outputs**

* **1**: ``attention_output`` - 2D tensor of type *T* (or the type set by ``set_out_type(0,…)``),
  shape ``[T, Hq*Sv]``.  Weighted sum of value vectors for each query token and each query head.

* **2**: ``score_aggregation`` - 1D tensor of type ``f32`` (or the type set by ``set_out_type(1,…)``),
  shape ``[T + sum(past_lens)]``.  Flat buffer of per-KV-position accumulated attention weights.
  The buffer is divided into one contiguous region per sequence; sequence ``s`` occupies
  positions ``[offset_s, offset_s + past_lens[s] + new_len_s)`` where ``offset_s`` is the
  sum of ``past_lens[r] + new_len_r`` for all ``r < s``.  Each position ``kpos`` holds the
  sum of softmax weights pointing from all qualifying query tokens (within the score aggregation window)
  and all ``Hq`` heads to that KV position.  This output is only meaningful when output port 1
  has downstream consumers; otherwise its buffer is zero-filled.

* **3**: ``diversity_scores`` - 1D tensor of type ``f32`` (or the type set by ``set_out_type(2,…)``),
  shape ``[sum_s(evictable_sizes[s]^2 / block_size)]``.  Flat diversity score matrix for
  adaptive KV eviction.  For sequence ``s`` with eviction zone of size ``E`` tokens and
  block size ``Bs``, the contribution is a flattened ``E/Bs × E`` matrix of pairwise
  block-level key diversity scores.  Output is absent (empty) when
  ``adaptive_rkv_evictable_sizes`` input is empty.


**Types**

* *T* - any supported real floating-point type (``f32``, ``f16``, ``bf16``).

* *T_real* - any supported real floating-point type; used for ``scale`` and ``alibi_slopes``.

* *T_cache* - cache precision; may differ from *T*.

  * Allowed for ``key_cache`` (input 4): ``u4``, ``i8``, ``u8``, ``f16``, ``f32``, ``bf16``.
  * Allowed for ``value_cache`` (input 5): ``u4``, ``u8``, ``f16``, ``f32``, ``bf16``.


**Dimension glossary**

* ``T``     - total new token count in the batch (``sum(new_len_s)``).
* ``B_seq`` - number of sequences in the batch.
* ``Hq``    - number of query heads.
* ``Hkv``   - number of KV heads (``Hkv`` divides ``Hq``; GQA group size = ``Hq / Hkv``).
* ``S``     - key / query head size.
* ``Sv``    - value head size (may equal ``S``).
* ``B``     - total number of physical blocks in the KV cache.
* ``Bs``    - tokens per block (block size).
* ``Nb``    - total entries in ``block_indices``.
* ``Nrot``  - number of blocks requiring RoPE re-rotation.
* ``C``     - number of rows in the RoPE trig LUT.
* ``Narkv`` - total entries in ``adaptive_rkv_diversity_block_set_indices``.

Output 0 dimension derivation:
``Hq * Sv = (Hq * S) * (Hkv * Sv) / (Hkv * S)``
i.e. ``query_features * value_features / key_features``.


**Shape inference**

* **Output 0**: dim 0 is copied from ``query`` dim 0; dim 1 = ``query_features * value_features / key_features``.  If any of these dimensions are dynamic the corresponding output dimension is dynamic.

* **Output 1**: if ``past_lens`` is a compile-time constant, shape = ``[new_token_count + sum(past_lens)]``; otherwise dynamic.

* **Output 2**: if ``adaptive_rkv_evictable_sizes`` is a compile-time constant and non-empty, shape = ``[max(evictable_sizes)]`` (approximation used during graph build; the exact size ``sum_s(evictable_sizes[s]^2 / block_size)`` is established at runtime).  Otherwise dynamic.


**Implementation notes**

The following behavioral differences exist between the reference implementation (TEMPLATE
plugin) and the CPU plugin kernel.  These are not specification violations - each
implementation is internally consistent - but they affect output values in specific
configurations.

1. **score_aggregation_window = 0 semantics**

   The integer value ``0`` in ``score_aggregation_window`` means *disabled* in both
   implementations: no attention weights are accumulated and output 1 is zero-filled
   for that sequence.  Positive values specify a window of the last ``N`` new tokens;
   negative values cause all new tokens to contribute (unbounded window).

2. **Output 2 (diversity scores) on CPU**

   The CPU plugin allocates the output 2 buffer with the correct size but does not compute
   its contents.  The buffer contains zeros after execution.  The ``AdaptiveRKVDiversityCalculator``
   that fills output 2 is implemented only in the reference.  Consumers of output 2 from
   the CPU plugin will receive a zero-filled buffer.

3. **adaptive_rkv_start_size (input 22) on CPU**

   The CPU kernel reads and stores this input but never forwards it to the executor.  Any
   semantics that rely on initial tokens being exempt from the eviction zone are not honored
   on the CPU path.

4. **Xattention activation conditions**

   Xattention (inputs 18-20) is only activated by the CPU kernel when **all** of the
   following are true: the platform is x86_64, the current step is prefill (``new_len > 1``),
   and the batch contains exactly one sequence (``B_seq == 1``).  In all other cases the CPU
   falls back to dense attention silently.  The reference implementation has no such restrictions.

5. **Sinks (input 21) on non-x86_64 platforms**

   The CPU plugin rejects a model that contains a non-empty sinks tensor on non-x86_64
   platforms (ARM64, etc.) at ``isSupportedOperation`` time.  The reference has no platform
   restriction.

6. **rotation_deltas (input 16) 2D form**

   The CPU executor requires ``rotation_deltas`` to have exactly 2 dimensions, with
   the second dimension being either ``1`` (per-block granularity) or ``Bs`` (per-token
   granularity).  Both forms are handled correctly in the CPU path.

   Two edge cases are not covered by the CPU implementation but are handled by the
   reference:

   * **1D input shape** (``[Nrot]``) — the spec allows a pure 1D tensor as a shorthand for
     per-block deltas.  The CPU executor's internal shape assertion requires 2D, so a
     1D tensor causes an assertion failure at runtime.  Pass ``[Nrot, 1]`` instead.

   * **Partial last block** — the CPU rotation kernel always iterates all ``Bs`` token
     slots in every block, including unwritten positions at the end of a partially-filled
     last block.  The reference only re-rotates positions up to ``past_len``, leaving
     unoccupied slots untouched.

7. **Cache statefulness and session lifetime**

   The ``PagedCacheManager`` is a heap-allocated object owned through a ``shared_ptr``
   stored on the op node.  Its lifetime is therefore tied to the lifetime of the compiled
   model: as long as any ``ov::CompiledModel`` or ``ov::InferRequest`` object referencing
   the model is alive, the manager (and all KV-cache data it holds) persists.

   When all compiled model and infer request handles are released, the reference count of
   the ``shared_ptr`` drops to zero, the manager is destroyed, and all cached KV data is
   freed.  The next call to ``compile_model`` with the same source model triggers
   ``AttachCacheManagerToPagedAttention`` again; it creates a brand-new, empty
   ``PagedCacheManager``, so the new inference session starts with a clean cache.

   There is no explicit ``reset`` API on the manager.  To start a fresh independent
   inference session without destroying the compiled model, the caller must attach a new
   manager via ``set_cache_manager`` before recompiling, or ensure the block table and
   ``past_lens`` inputs are reset to their initial state (all zeros / empty blocks) so the
   first ``infer`` call re-initializes the cache through the normal Phase 0 code path.

8. **Output 1 (score_aggregation) is informational only**

   Neither the CPU plugin nor the reference implementation modifies the KV cache based on
   the attention scores written to output 1.  The scores are computed and written to the
   output buffer; the op then returns without touching any cache block.

   All eviction decisions are the sole responsibility of the caller.  The intended workflow
   is:

   #. Read output 1 after each infer step.
   #. Determine which KV-cache blocks to evict (e.g. those with the lowest accumulated
      attention score).
   #. Update ``block_indices`` and ``past_lens`` on the subsequent ``infer`` call to reflect
      that those blocks are now free and should be overwritten.

   The op never evicts, copies, or invalidates any KV-cache block on its own.


**Known limitations**

* Output 2 is not computed by the CPU plugin.  It is only functional in the reference implementation.

* ``adaptive_rkv_start_size`` protection semantics are not honored by the CPU plugin.

* Xattention is limited to single-sequence prefill on x86_64 in the CPU plugin.

* Sinks are not supported on non-x86_64 platforms in the CPU plugin.

* 2D ``rotation_deltas`` are supported on both CPU and reference.  However, a pure 1D
  ``rotation_deltas`` tensor causes an assertion failure in the CPU executor; use shape
  ``[Nrot, 1]`` instead.  Partially-filled last blocks are not guarded: the CPU rotates
  all ``Bs`` slots unconditionally.


**CPU plugin KV-cache quantization properties**

The CPU plugin can quantize the KV cache at compile time.  The following properties are
passed as an ``ov::AnyMap`` to ``ov::Core::compile_model``; all are optional and the
defaults apply when omitted.

``KV_CACHE_PRECISION`` (C++: ``ov::hint::kv_cache_precision``, type: ``ov::element::Type``)
   Element type for both the key and value caches simultaneously.  Acts as an umbrella
   shorthand: it sets both ``KEY_CACHE_PRECISION`` and ``VALUE_CACHE_PRECISION`` unless
   either has already been supplied individually.

   Default: ``u8`` on x86_64 and ARM64; ``f16`` on all other platforms.

``KEY_CACHE_PRECISION`` (C++: ``ov::key_cache_precision``, type: ``ov::element::Type``)
   Element type for the key cache only.  Overrides the umbrella ``KV_CACHE_PRECISION``
   value for the key side.  Allowed types: ``u4``, ``i8``, ``u8``, ``f16``, ``bf16``,
   ``f32``.

``VALUE_CACHE_PRECISION`` (C++: ``ov::value_cache_precision``, type: ``ov::element::Type``)
   Element type for the value cache only.  Overrides the umbrella ``KV_CACHE_PRECISION``
   value for the value side.  Allowed types: ``u4``, ``u8``, ``f16``, ``bf16``, ``f32``.

``KEY_CACHE_GROUP_SIZE`` (C++: ``ov::key_cache_group_size``, type: ``uint64_t``)
   Sub-channel quantization group size for the key cache.  ``0`` (default) means the
   entire channel is treated as one group (no sub-channel grouping).  A non-zero value
   splits each channel into consecutive groups of that size before per-group
   scale/zero-point computation.

``VALUE_CACHE_GROUP_SIZE`` (C++: ``ov::value_cache_group_size``, type: ``uint64_t``)
   Sub-channel quantization group size for the value cache.  Same semantics as
   ``KEY_CACHE_GROUP_SIZE``.

``KEY_CACHE_QUANT_MODE`` (C++: ``ov::internal::key_cache_quant_mode``, type: ``CacheQuantMode``)
   Quantization granularity for the key cache:

   * ``AUTO``       - the plugin selects the granularity automatically.
   * ``BY_TOKEN``   - one scale/zero-point per token position.
   * ``BY_CHANNEL`` - one scale/zero-point per head-dimension channel.

   Default: ``AUTO``.

``VALUE_CACHE_QUANT_MODE`` (C++: ``ov::internal::value_cache_quant_mode``, type: ``CacheQuantMode``)
   Quantization granularity for the value cache.  Same values as
   ``KEY_CACHE_QUANT_MODE``.  Default: ``AUTO``.

``ENABLE_SAGE_ATTN`` (C++: ``ov::intel_cpu::enable_sage_attn``, type: ``bool``)
   Enables the SAGE (Scale Adaptive Group Efficient) attention execution path.  This
   path requires hardware support for AMX INT8 or VNNI2 instructions.  Default:
   ``false``.

.. note::

   These properties have no effect on the TEMPLATE (reference) plugin, which always
   processes key and value caches in the native floating-point type of the model.
   Non-``f32``/``f16`` cache precisions require the ``ConvertPagedAttnInputs``
   graph transformation to reinterpret the cache tensor layout before the kernel
   executes; this transformation is applied automatically by the CPU plugin during
   compilation.
