.. {#openvino_docs_ops_internal_Mamba2}

Mamba2
======


.. meta::
  :description: Learn about Mamba2 - a selective state-space sequence processing
                operation implementing the Mamba2 single-step recurrence.

**Versioned name**: *Mamba2*

**Category**: *Sequence processing*

**Short description**: *Mamba2* represents the selective state-space model (SSM)
recurrence used by Mamba2 mixers in hybrid Mamba2 models such as NemotronH.

**Detailed description**: *Mamba2* implements the Mamba2 selective state-space
recurrence (`arXiv:2405.21060 <https://arxiv.org/abs/2405.21060>`__). It updates an
SSM hidden state with a linear recurrence over the sequence dimension and reads out a
per-token output by contracting the state with the per-token output projection ``C``.

The discretized inputs ``dA`` (discretized state transition), ``dBx`` (discretized
input contribution ``B * x``) and ``C`` are precomputed and vectorized over the
sequence outside of this operation; *Mamba2* only performs the time-sequential state
recurrence and the readout. The single-step recurrence over the SSM state is a linear
recurrence:

.. math::

   state_t = state_{t-1} \cdot dA_t + dBx_t

   y_t = \sum_{n} state_t \cdot C_t

The skip connection ``x_t * D`` does not depend on the recurrent state and is therefore
added outside this operation. The following PyTorch-equivalent code illustrates the full
computation:

.. code-block:: py

   def torch_mamba2_recurrence(dA, dBx, C, recurrent_state):
       # dA:               [batch_size, num_heads, seq_len, 1, 1]
       # dBx:              [batch_size, num_heads, seq_len, head_dim, state_size]
       # C:                [batch_size, num_heads, seq_len, state_size]
       # recurrent_state:  [batch_size, num_heads, head_dim, state_size]
       batch_size, num_heads, seq_len, head_dim, state_size = dBx.shape

       output = torch.zeros(batch_size, num_heads, seq_len, head_dim).to(dBx)
       output_recurrent_state = recurrent_state

       for t in range(seq_len):
           dA_t = dA[:, :, t]      # [batch_size, num_heads, 1, 1] (broadcastable)
           dBx_t = dBx[:, :, t]    # [batch_size, num_heads, head_dim, state_size]
           C_t = C[:, :, t]        # [batch_size, num_heads, state_size]

           # state_t = state_{t-1} * dA_t + dBx_t
           output_recurrent_state = output_recurrent_state * dA_t + dBx_t

           # y_t = reduce_sum(state_t * C_t, axis=state_size) -> [batch_size, num_heads, head_dim]
           output[:, :, t] = (output_recurrent_state * C_t.unsqueeze(-2)).sum(dim=-1)

       return output, output_recurrent_state


**Attributes**

*Mamba2* operation has no attributes.


**Inputs**

* **1**: ``dA`` - 5D tensor of type *T* and shape
  ``[batch_size, num_heads, seq_len, 1, 1]``, the discretized state transition for each
  token and head. The two trailing singleton dimensions broadcast over ``head_dim`` and
  ``state_size`` when multiplying the recurrent state. **Required.**

* **2**: ``dBx`` - 5D tensor of type *T* and shape
  ``[batch_size, num_heads, seq_len, head_dim, state_size]``, the discretized input
  contribution (``B * x``) added to the recurrent state at each time step. **Required.**

* **3**: ``C`` - 4D tensor of type *T* and shape
  ``[batch_size, num_heads, seq_len, state_size]``, the per-token output projection used
  to read out the per-token output from the recurrent state. **Required.**

* **4**: ``recurrent_state`` - 4D tensor of type *T* and shape
  ``[batch_size, num_heads, head_dim, state_size]``, the recurrent (initially all-zeros)
  SSM hidden state. **Required.**


**Outputs**

* **1**: ``output`` - 4D tensor of type *T* and shape
  ``[batch_size, num_heads, seq_len, head_dim]``, the per-token output produced by
  contracting the updated state with ``C`` at each time step.

* **2**: ``output_recurrent_state`` - 4D tensor of type *T* and shape
  ``[batch_size, num_heads, head_dim, state_size]``, the SSM hidden state after
  processing the last token in the sequence.


**Types**

* *T*: any supported floating-point type.


**Example**

.. code-block:: xml
   :force:

   <layer ... type="Mamba2" ...>
       <input>
           <port id="0"> <!-- `dA` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>16</dim>
               <dim>1</dim>
               <dim>1</dim>
           </port>
           <port id="1"> <!-- `dBx` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>16</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
           <port id="2"> <!-- `C` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>16</dim>
               <dim>128</dim>
           </port>
           <port id="3"> <!-- `recurrent_state` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
       </input>
       <output>
           <port id="4"> <!-- `output` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>16</dim>
               <dim>64</dim>
           </port>
           <port id="5"> <!-- `output_recurrent_state` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>
