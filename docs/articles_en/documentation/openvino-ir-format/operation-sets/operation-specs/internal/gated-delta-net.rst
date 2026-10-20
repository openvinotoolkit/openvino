.. {#openvino_docs_ops_internal_GatedDeltaNet}

GatedDeltaNet
=============


.. meta::
  :description: Learn about GatedDeltaNet - a linear recurrent sequence processing
                operation based on the delta rule with a gating mechanism.

**Versioned name**: *GatedDeltaNet*

**Category**: *Sequence processing*

**Short description**: *GatedDeltaNet* represents a linear recurrent sequence model
that combines the delta rule memory update with a gating mechanism.

The optional snapshot mode was introduced for Qwen3.5 in the llama.cpp OpenVINO
backend, including the GGML ``qwen35`` and ``qwen35moe`` architectures used by
Qwen3.5 and Qwen3.6 models. During speculative MTP decoding, the target model
verifies a batch containing draft tokens. llama.cpp must restore the recurrent
matrix cache (``cache_s``) to the accepted token prefix when a draft is rejected;
the ordinary final-state output alone cannot provide that earlier state.
``snapshot_count`` exposes these prefix states without recomputing the recurrence.
Its default value of one preserves the existing final-state behavior.

**Detailed description**: *GatedDeltaNet* implements the recurrence from the paper
`arXiv:2412.06464 <https://arxiv.org/abs/2412.06464>`__. It processes a sequence of
query, key, and value vectors using the delta rule to update a hidden state matrix,
controlled by a per-token forget ``gate`` (applied as ``exp(g)``) and a per-token
write gate ``beta``. Queries are scaled by ``1 / sqrt(key_head_dim)`` before being used
to compute the output. The following PyTorch-equivalent code illustrates the full
computation:

.. code-block:: py

   def torch_recurrent_gated_delta_rule(
       query, key, value, recurrent_state, gate, beta,
       fuse_qk_l2norm=False, q_l2_norm_eps=1e-6, k_l2_norm_eps=1e-6, snapshot_count=1,
   ):
       def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
           """This function is intended to align with the l2norm implementation in the FLA library."""
           inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
           return x * inv_norm

       # Optional L2 normalization of query and key (when fuse_qk_l2norm is True)
       if fuse_qk_l2norm:
           query = l2norm(query, dim=-1, eps=q_l2_norm_eps)
           key = l2norm(key, dim=-1, eps=k_l2_norm_eps)

       num_value_heads = value.shape[2]
       groups = num_value_heads // key.shape[2]
       key = key.repeat_interleave(groups, dim=2)
       query = query.repeat_interleave(groups, dim=2)
       batch_size, sequence_length, num_heads, k_head_dim = key.shape
       v_head_dim = value.shape[-1]
       scale = 1 / (query.shape[-1] ** 0.5)
       query = query * scale

       output_attn = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)
       output_recurrent_state = recurrent_state
       snapshots = []

       for i in range(sequence_length):
           q_t = query[:, i]
           k_t = key[:, i]
           v_t = value[:, i]
           g_t = gate[:, i].exp().unsqueeze(-1).unsqueeze(-1)
           beta_t = beta[:, i].unsqueeze(-1)

           output_recurrent_state = output_recurrent_state * g_t
           kv_mem = (output_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
           delta = (v_t - kv_mem) * beta_t
           output_recurrent_state = output_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
           output_attn[:, i] = (output_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
           if snapshot_count > 1 and sequence_length - 1 - i < snapshot_count:
               snapshots.append(output_recurrent_state.clone())

       if snapshot_count > 1:
           # Only min(snapshot_count, sequence_length) entries are valid.
           snapshot_output = recurrent_state.new_empty((snapshot_count,) + recurrent_state.shape)
           if snapshots:
               snapshot_output[:len(snapshots)] = torch.stack(snapshots[::-1])
           return output_attn, snapshot_output
       return output_attn, output_recurrent_state


**Attributes**

* *fuse_qk_l2norm*

  * **Description**: When ``True``, applies L2 normalization to query and key vectors before
    the recurrent update, using ``q_l2_norm_eps`` and ``k_l2_norm_eps`` as the minimum
    normalization denominators to avoid division by zero.
  * **Type**: ``boolean``
  * **Default value**: ``False``
  * **Required**: *no*

* *q_l2_norm_eps*

  * **Description**: Epsilon value used as the minimum denominator when L2-normalizing query
    vectors. Only used when ``fuse_qk_l2norm`` is ``True``.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*

* *k_l2_norm_eps*

  * **Description**: Epsilon value used as the minimum denominator when L2-normalizing key
    vectors. Only used when ``fuse_qk_l2norm`` is ``True``.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*


* *snapshot_count*

  * **Description**: Number of newest-first recurrent state snapshots in output 1.
    A value of one preserves the four-dimensional final-state output. A larger
    value makes output 1 five-dimensional; it does not add a third output.
  * **Range of values**: positive integer
  * **Type**: ``int64``
  * **Default value**: ``1``
  * **Required**: *no*


**Inputs**

* **1**: ``query`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_heads, key_head_dim]``,
  the query vectors for each token and head. Scaled internally by ``1 / sqrt(key_head_dim)``
  before computing the output. **Required.**

* **2**: ``key`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_heads, key_head_dim]``,
  the key vectors for each token and head. **Required.**

* **3**: ``value`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, v_num_heads, value_head_dim]``,
  the value vectors for each token and head. **Required.**

* **4**: ``recurrent_state`` - 4D tensor of type *T* and shape
  ``[batch_size, v_num_heads, key_head_dim, value_head_dim]``, the recurrent (initially all-zeros) hidden state matrix.  **Required.**

* **5**: ``gate`` - 3D tensor of type *T* and shape ``[batch_size, seq_len, v_num_heads]``,
  the forget gate in log-space. Applied as ``exp(g)`` at each time step to decay the
  hidden state before the delta update. **Required.**

* **6**: ``beta`` - 3D tensor of type *T* and shape ``[batch_size, seq_len, v_num_heads]``,
  the write gate controlling how much of the delta correction is applied to the hidden
  state. **Required.**


**Outputs**

* **1**: ``output_attn`` - 4D tensor of type *T* and shape
  ``[batch_size, seq_len, v_num_heads, value_head_dim]``, the output vectors at each time step
  produced by applying the state matrix to the (scaled) query.

* **2**: ``output_recurrent_state`` - 4D tensor of type *T* and shape
  ``[batch_size, v_num_heads, key_head_dim, value_head_dim]``, the hidden state matrix
  after processing the last token in the sequence when ``snapshot_count=1``.
  When ``snapshot_count=K>1``, this output instead has shape
  ``[K,batch_size,v_num_heads,key_head_dim,value_head_dim]``. Entry ``i`` contains
  the state after zero-based token ``seq_len-1-i``. Entry 0 is the final state;
  only the first ``min(K,seq_len)`` entries are valid. Remaining entries are
  unspecified and must not be consumed. The state before token 0 is not included.


.. note::

   This operation uses grouped-query linear attention. The number of groups is
   ``num_groups = v_num_heads // num_heads``. Each query and key head is shared by
   ``num_groups`` consecutive value heads, with the mapping ``h_q = h_v // num_groups``.


**Snapshot mode and plugin support**

Snapshot mode supports llama.cpp Qwen3.5 speculative MTP rollback without
reevaluating the recurrent prefix. The GPU kernel leaves the input state unchanged and writes
newest-first states into output 1. The graph therefore has exactly two outputs
in both modes. For example, ``K=3`` and ``seq_len=2`` produce the states after
tokens 1 and 0 in entries 0 and 1; entry 2 is invalid.

The Intel GPU plugin supports snapshots. CPU and TEMPLATE implementations reject
``snapshot_count>1`` explicitly. Query/key normalization and grouped-query fusion
preserve the attribute. Paged GatedDeltaNet fusion skips snapshot mode because
its state-cache contract does not expose these snapshots.


**Types**

* *T*: any supported floating-point type.


**Example**

.. code-block:: xml
   :force:

   <layer ... type="GatedDeltaNet" ...>
       <input>
           <port id="0"> <!-- `query` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>64</dim>
           </port>
           <port id="1"> <!-- `key` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>64</dim>
           </port>
           <port id="2"> <!-- `value` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>128</dim>
           </port>
           <port id="3"> <!-- `recurrent_state` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
           <port id="4"> <!-- `gate` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
           </port>
           <port id="5"> <!-- `beta` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
           </port>
       </input>
       <output>
           <port id="6"> <!-- `output_attn` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>128</dim>
           </port>
           <port id="7"> <!-- `output_recurrent_state` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>
