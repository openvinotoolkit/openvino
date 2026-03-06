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
controlled by per-token forget (alpha) and write (beta) gates. Keys are L2-normalized
before being used to update the state.

.. code-block:: py

   GatedDeltaNet recurrence (applied independently per head, for each t = 1, ..., T):
     *    - matrix multiplication
    (.)   - Hadamard product (element-wise)
    (x)   - outer product
    ||.||_2 - L2 norm along the last (head_size) dimension

    # Normalize key
    k_norm_t = K_t / ||K_t||_2  # L2 norm along head_size dimension

    # Delta memory update: remove existing value associated with k_norm_t,
    # then write the new value, controlled by write gate beta_t
    S_t = alpha_t (.) S_{t-1} + beta_t * (V_t - S_{t-1} * k_norm_t) (x) k_norm_t

    # Compute output for each time step
    O_t = S_t * Q_t


**Inputs**

* **1**: ``Q`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_heads, head_size]``,
  the query vectors for each token and head. **Required.**

* **2**: ``K`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_heads, head_size]``,
  the key vectors for each token and head. Keys are L2-normalized internally before
  being used in the state update. **Required.**

* **3**: ``V`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_heads, head_size]``,
  the value vectors for each token and head. **Required.**

* **4**: ``beta`` - 4D tensor of type *T* and shape ``[batch_size, seq_len, num_heads, 1]``,
  the write gate (delta rule strength) controlling how much of the new value is written
  into the state. Expected values are in ``[0, 1]``. **Required.**

* **5**: ``alpha`` - 4D tensor of type *T* and shape
  ``[batch_size, seq_len, num_heads, head_size]``,
  the forget gate (per-element decay) applied to the state matrix before the delta
  update. Expected values are in ``[0, 1]``. **Required.**

* **6**: ``initial_state`` - 4D tensor of type *T* and shape
  ``[batch_size, num_heads, head_size, head_size]``, the initial hidden state matrix
  ``S_0``. If not provided, ``S_0`` is treated as an all-zeros matrix. **Optional.**


**Outputs**

* **1**: ``Y`` - 4D tensor of type *T* and shape
  ``[batch_size, seq_len, num_heads, head_size]``, the output vectors at each time step
  produced by applying the state matrix to the query.

* **2**: ``final_state`` - 4D tensor of type *T* and shape
  ``[batch_size, num_heads, head_size, head_size]``, the hidden state matrix
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
               <dim>64</dim>
           </port>
           <port id="3"> <!-- `beta` write gate -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>1</dim>
           </port>
           <port id="4"> <!-- `alpha` forget gate -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>64</dim>
           </port>
           <port id="5"> <!-- `initial_state` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>64</dim>
           </port>
       </input>
       <output>
           <port id="6"> <!-- `Y` output -->
               <dim>1</dim>
               <dim>16</dim>
               <dim>8</dim>
               <dim>64</dim>
           </port>
           <port id="7"> <!-- `final_state` -->
               <dim>1</dim>
               <dim>8</dim>
               <dim>64</dim>
               <dim>64</dim>
           </port>
       </output>
   </layer>
