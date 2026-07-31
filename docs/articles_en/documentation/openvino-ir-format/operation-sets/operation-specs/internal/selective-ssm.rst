.. {#openvino_docs_ops_internal_SelectiveSSM}

SelectiveSSM
============


.. meta::
  :description: Learn about SelectiveSSM - a selective state-space model (selective scan)
                sequence processing operation implementing the Mamba2 recurrence.

**Versioned name**: *SelectiveSSM*

**Category**: *Sequence processing*

**Short description**: *SelectiveSSM* (selective state-space model) represents the selective
state-space recurrence used by Mamba2 mixers in hybrid Mamba2 models such as
NemotronH.

**Detailed description**: *SelectiveSSM* implements the Mamba2 selective state-space
recurrence (`arXiv:2405.21060 <https://arxiv.org/abs/2405.21060>`__). It updates an
SSM hidden state with a linear recurrence over the sequence dimension and reads out a
per-token output by contracting the state with the per-token output projection ``C``.

The operation takes the raw (un-discretized) parameters ``A`` (log-decay rates), ``dt``
(time steps), ``B`` (input projection), ``x`` (input hidden states) and ``C`` (output
projection). Discretization is performed inside the operation, per time step, following
the standard Mamba2 discretization:

.. math::

   dA_t = \exp(A \cdot dt_t)

   dBx_t = (dt_t \cdot B_t) \otimes x_t

   state_t = state_{t-1} \cdot dA_t + dBx_t

   y_t = \sum_{n} state_t \cdot C_t

The ``B`` and ``C`` projections are grouped: they are shared across heads and expanded
from ``num_groups`` to ``num_heads`` by repeating each group ``num_heads // num_groups``
times before the recurrence. The skip connection ``x_t * D`` does not depend on the
recurrent state and is therefore added outside this operation. The following
PyTorch-equivalent code illustrates the full (grouped) computation:

.. code-block:: py

   def torch_selective_ssm_recurrence(A, dt, B, x, C, recurrent_state):
       # A:               [num_heads]
       # dt:              [batch_size, seq_len, num_heads]
       # B:               [batch_size, seq_len, num_groups, state_size]
       # x:               [batch_size, seq_len, num_heads, head_dim]
       # C:               [batch_size, seq_len, num_groups, state_size]
       # recurrent_state: [batch_size, num_heads, head_dim, state_size]
       batch_size, seq_len, num_heads, head_dim = x.shape
       num_groups = B.shape[2]
       heads_per_group = num_heads // num_groups

       # Expand the grouped B/C projections from num_groups to num_heads
       # by repeating each group heads_per_group times.
       B = B.repeat_interleave(heads_per_group, dim=2)  # [batch_size, seq_len, num_heads, state_size]
       C = C.repeat_interleave(heads_per_group, dim=2)  # [batch_size, seq_len, num_heads, state_size]

       output = torch.zeros(batch_size, seq_len, num_heads, head_dim).to(x)
       output_recurrent_state = recurrent_state

       for t in range(seq_len):
           dt_t = dt[:, t]     # [batch_size, num_heads]
           B_t = B[:, t]       # [batch_size, num_heads, state_size]
           x_t = x[:, t]       # [batch_size, num_heads, head_dim]
           C_t = C[:, t]       # [batch_size, num_heads, state_size]

           # Discretization
           dA_t = torch.exp(A * dt_t)                                            # [batch_size, num_heads]
           dBx_t = (dt_t.unsqueeze(-1) * B_t).unsqueeze(-2) * x_t.unsqueeze(-1)  # [batch_size, num_heads, head_dim, state_size]

           # state_t = state_{t-1} * dA_t + dBx_t
           output_recurrent_state = output_recurrent_state * dA_t.unsqueeze(-1).unsqueeze(-1) + dBx_t

           # y_t = reduce_sum(state_t * C_t, axis=state_size) -> [batch_size, num_heads, head_dim]
           output[:, t] = (output_recurrent_state * C_t.unsqueeze(-2)).sum(dim=-1)

       return output, output_recurrent_state


**Attributes**

*SelectiveSSM* operation has no attributes.


**Inputs**

* **1**: ``A`` - 1D tensor of type *T* and shape ``[num_heads]``, the (negative) log-decay
  rates per head used to compute the discretized state transition ``dA``. **Required.**

* **2**: ``dt`` - 3D tensor of type *T* and shape ``[batch_size, seq_len, num_heads]``, the
  per-token, per-head time steps used for discretization. **Required.**

* **3**: ``B`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_groups, state_size]``,
  the grouped input projection. Expanded from ``num_groups`` to ``num_heads`` inside the
  operation. **Required.**

* **4**: ``x`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_heads, head_dim]``,
  the input hidden states. **Required.**

* **5**: ``C`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_groups, state_size]``,
  the grouped output projection. Expanded from ``num_groups`` to ``num_heads`` inside the
  operation. **Required.**

* **6**: ``recurrent_state`` - 4D tensor of type *T* and shape
  ``[batch_size, num_heads, head_dim, state_size]``, the recurrent (initially all-zeros)
  SSM hidden state. **Required.**


**Outputs**

* **1**: ``output`` - 4D tensor of type *T* and shape
  ``[batch_size, seq_len, num_heads, head_dim]``, the per-token output produced by
  contracting the updated state with ``C`` at each time step.

* **2**: ``output_recurrent_state`` - 4D tensor of type *T* and shape
  ``[batch_size, num_heads, head_dim, state_size]``, the SSM hidden state after
  processing the last token in the sequence.


.. note::

   This operation uses grouped projections. The number of heads sharing each group is
   ``heads_per_group = num_heads // num_groups``. Each group of ``B``/``C`` is repeated
   ``heads_per_group`` times to expand it to ``num_heads`` before the recurrence.


**Types**

* *T*: any supported floating-point type.


**Example**

.. code-block:: xml
   :force:

   <layer ... type="SelectiveSSM" ...>
       <input>
           <port id="0"> <!-- `A` -->
               <dim>8</dim>
           </port>
           <port id="1"> <!-- `dt` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
           </port>
           <port id="2"> <!-- `B` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>2</dim>
               <dim>128</dim>
           </port>
           <port id="3"> <!-- `x` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>64</dim>
           </port>
           <port id="4"> <!-- `C` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>2</dim>
               <dim>128</dim>
           </port>
           <port id="5"> <!-- `recurrent_state` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
       </input>
       <output>
           <port id="6"> <!-- `output` -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>64</dim>
           </port>
           <port id="7"> <!-- `output_recurrent_state` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>
