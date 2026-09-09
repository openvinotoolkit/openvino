.. {#openvino_docs_ops_internal_SelectiveSSM}

SelectiveSSM
============


.. meta::
  :description: Learn about SelectiveSSM - a selective state-space model (selective scan)
                sequence processing operation implementing the Mamba2 recurrence.

**Versioned name**: *SelectiveSSM*

**Category**: *Internal*

**Short description**: *SelectiveSSM* (selective state-space model) represents the selective
state-space recurrence used by Mamba2 mixers in hybrid Mamba2 models such as
NemotronH.

**Detailed description**: *SelectiveSSM* implements the Mamba2 selective state-space
recurrence (`arXiv:2405.21060 <https://arxiv.org/abs/2405.21060>`__). It updates an
SSM hidden state with a linear recurrence over the sequence dimension and reads out a
per-token output by contracting the state with the per-token output projection ``C``.

The operation takes the raw (un-discretized) parameters ``A`` (log-decay rates), ``dt``
(time steps), ``B`` (input projection), ``x`` (input hidden states) and ``C`` (output
projection). The time discretization of ``A`` and ``B`` is performed vectorized over the
whole sequence before the recurrence (cheaper than recomputing it per time step):

.. math::

   dA = \exp(A \cdot dt)

   dtB = dt \cdot B

The per-timestep recurrence then only forms the outer product :math:`dtB_t \otimes x_t`
(its full ``[batch_size, seq_len, num_heads, head_dim, state_size]`` form is too large to
materialize up front):

.. math::

   dBx_t = dtB_t \otimes x_t

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

       # Vectorized time discretization of A and B over the whole sequence.
       dA = torch.exp(A * dt).reshape(batch_size, seq_len, num_heads, 1, 1)
       dB = dt.reshape(batch_size, seq_len, num_heads, 1) * B
       # dBx shape - [batch_size, seq_len, num_heads, head_dim, state_size]
       dBx = dB.reshape(batch_size, seq_len, num_heads, 1, -1) * x.reshape(batch_size, seq_len, num_heads, head_dim, 1)
       C = C.reshape(batch_size, seq_len, num_heads, 1, -1)

       output_recurrent_state = recurrent_state
       output = torch.zeros(batch_size, seq_len, num_heads, head_dim).to(x)

       for t in range(seq_len):
           output_recurrent_state = output_recurrent_state * dA[:, t] + dBx[:, t]
           output[:, t] = (output_recurrent_state * C[:, t]).sum(dim=-1)

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

* *T*: any floating-point type.


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
