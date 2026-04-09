GroupedMatMul
=============


.. meta::
  :description: Learn about GroupedMatMul-17 - a grouped matrix multiplication operation
                for Mixture of Experts (MoE) workloads.

**Versioned name**: *GroupedMatMul-17*

**Category**: *Matrix multiplication*

**Short description**: Grouped matrix multiplication for Mixture of Experts (MoE).

**Detailed description**

*GroupedMatMul* operation computes multiple matrix multiplications where each group processes
a subset of the input data. This operation is optimized for Mixture of Experts (MoE) workloads
where tokens are routed to different expert networks.

The operation supports three input combinations based on tensor dimensions:

**Case 1: 2D × 3D (MoE Forward Pass)**

Used for computing expert outputs from routed tokens:

* ``mat_a``: Shape ``[total_tokens, K]`` - all tokens packed contiguously
* ``mat_b``: Shape ``[G, K, N]`` - per-group (expert) weight matrices
* ``offsets``: Shape ``[G]`` - cumulative token boundaries

For group ``i``, computes: ``output[start:end] = mat_a[start:end] @ mat_b[i]``

where ``start = offsets[i-1]`` (or 0 for i=0) and ``end = offsets[i]``.

Output shape: ``[total_tokens, N]``

**Case 2: 3D × 3D (Batched Uniform Groups)**

Used when all groups have the same number of tokens:

* ``mat_a``: Shape ``[G, M, K]`` - per-group inputs
* ``mat_b``: Shape ``[G, K, N]`` - per-group weights
* ``offsets``: Not used (must not be provided)

For each group ``i``, computes: ``output[i] = mat_a[i] @ mat_b[i]``

Output shape: ``[G, M, N]``

**Case 3: 2D × 2D (MoE Weight Gradient)**

Used during backpropagation for computing per-expert weight gradients:

* ``mat_a``: Shape ``[K, total_tokens]`` - transposed activations
* ``mat_b``: Shape ``[total_tokens, N]`` - gradient output
* ``offsets``: Shape ``[G]`` - cumulative token boundaries

For group ``i``, computes: ``output[i] = mat_a[:, start:end] @ mat_b[start:end, :]``

Output shape: ``[G, K, N]``

**Offsets Format**

The ``offsets`` tensor contains cumulative token counts. For G groups:
``offsets = [M0, M0+M1, M0+M1+M2, ..., total_tokens]``

where ``Mi`` is the number of tokens for group ``i``.

For example, with tokens per group ``[3, 5, 2]``, offsets would be ``[3, 8, 10]``.

**Attributes**

*GroupedMatMul* operation has no attributes.

**Inputs**

* **1**: ``mat_a`` - Tensor of type *T* with first operand. Required.
  
  * Case 1 (2D×3D): Shape ``[total_tokens, K]``
  * Case 2 (3D×3D): Shape ``[G, M, K]``
  * Case 3 (2D×2D): Shape ``[K, total_tokens]``

* **2**: ``mat_b`` - Tensor of type *T* with second operand. Required.
  
  * Case 1 (2D×3D): Shape ``[G, K, N]``
  * Case 2 (3D×3D): Shape ``[G, K, N]``
  * Case 3 (2D×2D): Shape ``[total_tokens, N]``

* **3**: ``offsets`` - 1D tensor of type *T_IDX* with group boundaries. Optional.
  
  * Shape: ``[G]`` containing cumulative offsets
  * Required for Case 1 (2D×3D) and Case 3 (2D×2D)
  * Must not be provided for Case 2 (3D×3D)

**Outputs**

* **1**: Tensor of type *T* with matrix multiplication results.
  
  * Case 1 (2D×3D): Shape ``[total_tokens, N]``
  * Case 2 (3D×3D): Shape ``[G, M, N]``
  * Case 3 (2D×2D): Shape ``[G, K, N]``

**Types**

* *T*: any supported floating-point or integer type.
* *T_IDX*: ``int32`` or ``int64``.

**Example**

*MoE Forward Pass (Case 1: 2D × 3D)*

.. code-block:: xml
   :force:

   <layer ... type="GroupedMatMul" version="opset17">
       <input>
           <port id="0">  <!-- mat_a: 10 tokens, K=64 -->
               <dim>10</dim>
               <dim>64</dim>
           </port>
           <port id="1">  <!-- mat_b: 3 experts, K=64, N=128 -->
               <dim>3</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
           <port id="2">  <!-- offsets: [3, 8, 10] -->
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="3">  <!-- output: 10 tokens, N=128 -->
               <dim>10</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>

*Batched Uniform (Case 2: 3D × 3D)*

.. code-block:: xml
   :force:

   <layer ... type="GroupedMatMul" version="opset17">
       <input>
           <port id="0">  <!-- mat_a: 3 groups, M=4, K=64 -->
               <dim>3</dim>
               <dim>4</dim>
               <dim>64</dim>
           </port>
           <port id="1">  <!-- mat_b: 3 groups, K=64, N=128 -->
               <dim>3</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
       </input>
       <output>
           <port id="2">  <!-- output: 3 groups, M=4, N=128 -->
               <dim>3</dim>
               <dim>4</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>

*Weight Gradient (Case 3: 2D × 2D)*

.. code-block:: xml
   :force:

   <layer ... type="GroupedMatMul" version="opset17">
       <input>
           <port id="0">  <!-- mat_a: K=64, total_tokens=16 -->
               <dim>64</dim>
               <dim>16</dim>
           </port>
           <port id="1">  <!-- mat_b: total_tokens=16, N=128 -->
               <dim>16</dim>
               <dim>128</dim>
           </port>
           <port id="2">  <!-- offsets: [4, 12, 16] for 3 groups -->
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="3">  <!-- output: 3 groups, K=64, N=128 -->
               <dim>3</dim>
               <dim>64</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>
