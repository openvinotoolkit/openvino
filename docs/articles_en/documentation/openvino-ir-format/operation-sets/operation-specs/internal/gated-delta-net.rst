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

**Detailed description**: *GatedDeltaNet* implements the recurrence from the paper
`arXiv:2412.06464 <https://arxiv.org/abs/2412.06464>`__. It processes a sequence of
query, key, and value vectors using the delta rule to update a hidden state matrix,
controlled by a per-token forget gate ``g`` (applied as ``exp(g)``) and a per-token
write gate ``beta``. Queries are scaled by ``1 / sqrt(head_size)`` before being used
to compute the output. The following PyTorch-equivalent code illustrates the full
computation:

.. code-block:: py

   def torch_recurrent_gated_delta_rule(
       query, key, value, g, beta, initial_state,
   ):
       batch_size, sequence_length, num_heads, k_head_dim = key.shape
       v_head_dim = value.shape[-1]
       scale = 1 / (query.shape[-1] ** 0.5)
       query = query * scale

       core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)
       last_recurrent_state = initial_state

       for i in range(sequence_length):
           q_t = query[:, i]
           k_t = key[:, i]
           v_t = value[:, i]
           g_t = g[:, i].exp().unsqueeze(-1).unsqueeze(-1)
           beta_t = beta[:, i].unsqueeze(-1)

           last_recurrent_state = last_recurrent_state * g_t
           kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
           delta = (v_t - kv_mem) * beta_t
           last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
           core_attn_out[:, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

       return core_attn_out, last_recurrent_state


**Inputs**

* **1**: ``Q`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_heads, key_head_dim]``,
  the query vectors for each token and head. Scaled internally by ``1 / sqrt(key_head_dim)``
  before computing the output. **Required.**

* **2**: ``K`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_heads, key_head_dim]``,
  the key vectors for each token and head. **Required.**

* **3**: ``V`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_heads, value_head_dim]``,
  the value vectors for each token and head. **Required.**

* **4**: ``g`` - 3D tensor of type *T* and shape ``[batch_size, seq_len, num_heads]``,
  the forget gate in log-space. Applied as ``exp(g)`` at each time step to decay the
  hidden state before the delta update. **Required.**

* **5**: ``beta`` - 3D tensor of type *T* and shape ``[batch_size, seq_len, num_heads]``,
  the write gate controlling how much of the delta correction is applied to the hidden
  state. **Required.**

* **6**: ``initial_state`` - 4D tensor of type *T* and shape
  ``[batch_size, num_heads, key_head_dim, value_head_dim]``, the initial hidden state matrix.
  If not provided, the initial state is treated as an all-zeros matrix. **Optional.**


**Outputs**

* **1**: ``Y`` - 4D tensor of type *T* and shape
  ``[batch_size, seq_len, num_heads, value_head_dim]``, the output vectors at each time step
  produced by applying the state matrix to the (scaled) query.

* **2**: ``final_state`` - 4D tensor of type *T* and shape
  ``[batch_size, num_heads, key_head_dim, value_head_dim]``, the hidden state matrix
  after processing the last token in the sequence.


**Types**

* *T*: any supported floating-point type.


**Example**

.. code-block:: xml
   :force:

   <layer ... type="GatedDeltaNet" ...>
       <input>
           <port id="0"> <!-- `Q` query -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>64</dim>
           </port>
           <port id="1"> <!-- `K` key -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>64</dim>
           </port>
           <port id="2"> <!-- `V` value -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>128</dim>
           </port>
           <port id="3"> <!-- `g` log forget gate -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
           </port>
           <port id="4"> <!-- `beta` write gate -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
           </port>
           <port id="5"> <!-- `initial_state` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
       </input>
       <output>
           <port id="6"> <!-- `Y` output -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>128</dim>
           </port>
           <port id="7"> <!-- `final_state` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>64</dim>
           </port>
       </output>
   </layer>
