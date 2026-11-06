.. {#openvino_docs_ops_internal_GatedResidual}

GatedResidual
=============

.. meta::
  :description: Learn about GatedResidual, an internal operation for branch-wise gated residual READ, WRITE, and FINAL_MIX computations used by hyper-connection transformer models.

**Versioned name**: *GatedResidual*

**Category**: *Internal*

**Short description**:
The *GatedResidual* operation represents the gated residual (hyper-connection) mechanism used by Qwen4-Exp / Qwen3.8-Flash-Next-style models. It handles branch-wise normalization and gated branch merging, the gated residual update, and an optional terminal merge. READ and WRITE are distinct operation modes; the sub-layer output is an explicit input to WRITE. READ and FINAL_MIX optionally accept ``i4``/``u4``/``i8``/``u8`` compressed ``w_down`` / ``w_up`` / ``w_inject`` weights.

**Detailed description**

A residual stream with ``C`` branches and per-branch hidden size ``D`` is physically stored as
``[B, T, C*D]`` and logically interpreted as ``[B, T, C, D]``. ``GatedResidual`` provides three
modes:

* ``READ`` normalizes each branch independently, computes a low-rank channel-wise read gate,
  merges the gated branches, and may also emit a per-branch write gate.
* ``WRITE`` broadcasts a previously computed per-branch write gate over the hidden dimension
  and adds the gated sub-layer output to every residual branch.
* ``FINAL_MIX`` performs the READ merge without producing a write gate, for a terminal branch
  mixer.

A typical transformer region is:

.. code-block:: text

   GatedResidual(READ) -> attention / MLP / other sub-layer -> GatedResidual(WRITE)

The branch normalization is over each branch's ``D`` channels, not over the concatenated
``C*D`` dimension. The factors ``1/C`` in the down projection, branch merge, and write-gate
projection are part of the operation semantics. The read gate is per branch and channel; the
write gate is one scalar per branch and token, with range ``(0, 2)``.

**Mathematical definition**

Let ``R_c`` be branch ``c`` of the residual row, ``Rbar_c`` its normalized value, ``W_down``
shape ``[L, C*D]``, ``W_up`` shape ``[C*D, L]``, and ``W_inject`` shape ``[C, C*D]``. The
stored normalization weight ``w`` yields the gain :math:`\gamma = 1 + w` when
``norm_gain_plus_one`` is true; otherwise :math:`\gamma = w`. :math:`\gamma_c` is the
``D``-element segment of :math:`\gamma` that belongs to branch ``c``.

For each branch:

.. math::

   \bar{R}_c = \frac{R_c}{\sqrt{\operatorname{mean}(R_c^2) + \epsilon}} \odot \gamma_c

where the mean is taken over the ``D`` channels of branch ``c``.

For the READ projections, let ``Rbar`` denote the normalized branches represented as a
``C*D``-wide vector:

.. math::

   T = \operatorname{SiLU}\left(\frac{\bar{R} W_{down}^{T}}{C}\right)

   G = \operatorname{sigmoid}(T W_{up}^{T})

   M = \frac{1}{C}\sum_{c=0}^{C-1} G_c \odot \bar{R}_c

``G`` is a per-branch, per-channel gate. If ``produce_write_gate`` is true, READ also computes:

.. math::

   S = 2\,\operatorname{sigmoid}\left(\frac{\bar{R} W_{inject}^{T}}{C}\right)

``S`` has logical shape ``[B, T, C]``. Its values are in ``(0, 2)`` and it must not be clamped
to ``(0, 1)``.

WRITE mode uses the original widened residual and the sub-layer output ``Y``:

.. math::

   R'_c = R_c + S_c Y

The scalar ``S_c`` broadcasts across the ``D`` channels of branch ``c``. ``FINAL_MIX`` uses
the READ equations for ``M`` but does not calculate ``S``.

The following PyTorch-equivalent code illustrates the full computation. Weights are given as
decompressed floating-point tensors; ``dequantize_weight`` shows how the compressed
``i4``/``u4``/``i8``/``u8`` weights (``w_down``, ``w_up`` and ``w_inject``) are expanded first, with ``zp``
already unpacked to one value per element (signed types sign-extended) or ``None`` when the op has no zero points.

.. code-block:: py

   import torch
   import torch.nn.functional as F

   def dequantize_weight(q, scale, zp=None):
       # q: [N, K] i4/u4/i8/u8 values, scale / zp: [G, N]; the group size K // G is derived from the shapes
       N, K = q.shape
       G = scale.shape[0]
       assert K % G == 0
       s = scale.t().repeat_interleave(K // G, dim=1)   # [N, K]
       z = zp.t().repeat_interleave(K // G, dim=1).float() if zp is not None else 0.0   # no zero points: 0
       return ((q.float() - z) * s.float()).half()

   def branch_rms_norm(residual, norm_weight, C, D, epsilon, norm_gain_plus_one):
       # residual: [..., C*D]; every branch is normalized over its own D channels
       r = residual.float().unflatten(-1, (C, D))
       r = r * torch.rsqrt(r.pow(2).mean(dim=-1, keepdim=True) + epsilon)
       gain = norm_weight.float().unflatten(-1, (C, D))
       if norm_gain_plus_one:
           gain = 1.0 + gain
       return r * gain                                  # [..., C, D]

   def gated_residual_read(residual, norm_weight, w_down, w_up, w_inject,
                           C, D, epsilon=1e-6, norm_gain_plus_one=True,
                           produce_write_gate=False):
       # READ (produce_write_gate may be True) and FINAL_MIX (w_inject is None)
       r_bar = branch_rms_norm(residual, norm_weight, C, D, epsilon, norm_gain_plus_one)
       r_bar_flat = r_bar.flatten(-2)                   # [..., C*D]

       t = F.silu(F.linear(r_bar_flat, w_down.float()) / C)             # [..., L]
       gate = torch.sigmoid(F.linear(t, w_up.float())).unflatten(-1, (C, D))
       mixed = (gate * r_bar).mean(dim=-2)                              # [..., D]

       if not produce_write_gate:
           return mixed.to(residual.dtype)
       write_gate = 2.0 * torch.sigmoid(F.linear(r_bar_flat, w_inject.float()) / C)  # [..., C]
       return mixed.to(residual.dtype), write_gate.to(residual.dtype)

   def gated_residual_write(residual, sublayer_y, write_gate, C, D):
       # residual: [..., C*D], sublayer_y: [..., D], write_gate: [..., C] in (0, 2)
       r = residual.float().unflatten(-1, (C, D))
       out = r + write_gate.float().unsqueeze(-1) * sublayer_y.float().unsqueeze(-2)
       return out.flatten(-2).to(residual.dtype)        # [..., C*D]

**Attributes**

* *mode*

  * **Description**: Selects ``READ``, ``WRITE``, or ``FINAL_MIX``.
  * **Range of values**: ``0`` (READ), ``1`` (WRITE), ``2`` (FINAL_MIX)
  * **Type**: ``int``
  * **Required**: *yes*

* *hc_count*

  * **Description**: Number of residual branches, :math:`C`.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *hidden_size*

  * **Description**: Hidden width of each branch, :math:`D`.
  * **Range of values**: a positive integer
  * **Type**: ``int``
  * **Required**: *yes*

* *lowrank*

  * **Description**: Low-rank width, :math:`L`, used by READ and FINAL_MIX projections. It must be positive for READ and FINAL_MIX and is unused by WRITE.
  * **Range of values**: a positive integer for READ / FINAL_MIX; any non-negative integer for WRITE
  * **Type**: ``int``
  * **Required**: *yes*

* *epsilon*

  * **Description**: Epsilon used by branch-wise RMS normalization.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*

* *norm_gain_plus_one*

  * **Description**: If true, interpret the stored normalization weight as ``1 + w``; otherwise use it as the gain directly.
  * **Type**: ``bool``
  * **Default value**: ``true``
  * **Required**: *no*

* *produce_write_gate*

  * **Description**: In READ mode, compute and emit the write gate using ``w_inject``. It must be false for FINAL_MIX and is unused in WRITE mode.
  * **Type**: ``bool``
  * **Default value**: ``false``
  * **Required**: *no*

**Inputs**

Input indices depend on ``mode``. All dimensions below use ``B`` for batch, ``T`` for tokens,
``C`` for branches, ``D`` for per-branch hidden width, and ``L`` for low-rank width. The logical
residual shape is ``[B, T, C, D]``; the physical shape is ``[B, T, C*D]``. Any rank of at least 2
is accepted for the data inputs (``residual``, ``sublayer_y``, ``write_gate``), as long as the
last dimension is as specified below and the leading dimensions agree across them.

* **READ / FINAL_MIX input 0**: ``residual``
  A floating-point tensor with logical shape ``[B, T, C, D]`` and physical shape
  ``[B, T, C*D]``. **Required.**

* **READ / FINAL_MIX input 1**: ``norm_weight``
  A 1D tensor of shape ``[C*D]`` containing the branch-wise normalization gains in flattened
  storage order. Each branch uses its own contiguous ``D``-element segment. **Required.**

* **READ / FINAL_MIX input 2**: ``w_down``
  A 2D tensor of shape ``[L, C*D]`` for the low-rank down projection. Floating-point in the
  uncompressed forms, ``i4``, ``u4``, ``i8`` or ``u8`` in the compressed forms. **Required.**

* **READ / FINAL_MIX input 3**: ``w_up``
  A 2D tensor of shape ``[C*D, L]`` for the up projection that produces read-gate logits. Floating-point in the
  uncompressed forms, ``i4``, ``u4``, ``i8`` or ``u8`` in the compressed forms. **Required.**

READ and FINAL_MIX accept one of six input forms, selected by the weight element type, by whether
zero points are present, and by ``produce_write_gate``. Inputs 0-3 are identical in all of them. The
compression parameters, when present, follow input 3. The write-gate inputs, when present, are always
the **last** inputs, so the index of ``w_inject`` depends on the form:

.. list-table::
   :header-rows: 1

   * - Form
     - Inputs
   * - Floating-point weights
     - 4 inputs: ``residual``, ``norm_weight``, ``w_down``, ``w_up``
   * - Floating-point weights, write gate
     - 5 inputs: inputs 0-3 as above, then ``w_inject`` (input 4)
   * - Compressed weights, no zero points
     - 6 inputs: inputs 0-3 as above, then ``scale_down`` (4), ``scale_up`` (5)
   * - Compressed weights
     - 8 inputs: inputs 0-5 as in the previous form, then ``zp_down`` (6), ``zp_up`` (7)
   * - Compressed weights, no zero points, write gate
     - 8 inputs: inputs 0-5 as in the compressed form without zero points, then ``w_inject`` (6),
       ``scale_inject`` (7)
   * - Compressed weights, write gate
     - 11 inputs: inputs 0-7 as in the compressed form, then ``w_inject`` (8), ``scale_inject`` (9),
       ``zp_inject`` (10)

Zero points are optional but all-or-nothing: either ``zp_down`` and ``zp_up`` (and ``zp_inject``) are all
present or none is, and a missing zero point is 0. For a given ``produce_write_gate`` the number of
inputs selects the form unambiguously.

* **READ / FINAL_MIX inputs 4-5 (compressed forms only)**: when the weights are ``i4``, ``u4``, ``i8`` or
  ``u8``, these inputs carry the scales of ``w_down`` and ``w_up``.

  * input 4: ``scale_down``, ``f16``, shape ``[G_d, L]``
  * input 5: ``scale_up``, ``f16``, shape ``[G_u, C*D]``

* **READ / FINAL_MIX inputs 6-7 (compressed forms with zero points only)**:

  * input 6: ``zp_down``, same element type as ``w_down``, same shape as ``scale_down``
  * input 7: ``zp_up``, same element type as ``w_up``, same shape as ``scale_up``

* **READ write-gate inputs (when produce_write_gate is true; always the last inputs)**:

  * ``w_inject``: a 2D tensor of shape ``[C, C*D]`` used to compute the write gate. Floating-point
    in the five-input form (input 4), ``i4``, ``u4``, ``i8`` or ``u8`` in the compressed forms
    (input 6 without zero points, input 8 with them).
  * ``scale_inject``, compressed forms only: ``f16``, shape ``[G_i, C]`` (the input after ``w_inject``).
  * ``zp_inject``, compressed form with zero points only (input 10): same element type as ``w_inject``,
    same shape as ``scale_inject``.

For every compressed weight, scales and zero points are laid out ``[G, N]`` with ``N`` (the output
channel) contiguous. The group size is not an attribute: it is derived as ``group_size = K / G``,
where ``K`` is the number of columns of the weight (``C*D`` for ``w_down`` and ``w_inject``, ``L``
for ``w_up``) and ``G`` the number of rows of the scale and zero point. ``K`` must be divisible by
``G``; ``G = 1`` means per-channel quantization. ``w_down``, ``w_up`` and ``w_inject`` (when
present) must all share one element type; ``i4``/``u4``/``i8``/``u8`` weights are accepted only in the compressed
forms. 4-bit weights and zero points are packed two elements per byte in row-major order, the even
element in the low nibble; ``i4`` elements are two's complement nibbles. Dequantization is
``w = (q - zp) * scale`` with ``q`` and ``zp`` interpreted according to the element type, see ``dequantize_weight``
above.

* **WRITE input 0**: ``residual``
  A floating-point tensor with logical shape ``[B, T, C, D]`` and physical shape
  ``[B, T, C*D]``. **Required.**

* **WRITE input 1**: ``sublayer_y``
  A floating-point tensor of shape ``[B, T, D]``, containing the output of the intervening
  sub-layer. **Required.**

* **WRITE input 2**: ``write_gate``
  A floating-point tensor of shape ``[B, T, C]``. The gate is scalar per branch and broadcasts
  over that branch's ``D`` channels. **Required.**

**Outputs**

* **READ / FINAL_MIX output 0**: ``mixed``
  A tensor of shape ``[B, T, D]``. It is the mean of the branch-wise, read-gated normalized
  residual states.

* **READ output 1**: ``write_gate``
  Present only when ``produce_write_gate`` is true. Shape ``[B, T, C]``; values are in ``(0, 2)``.

* **WRITE output 0**: ``residual_out``
  A tensor with the same shape as the residual input, ``[B, T, C*D]`` (logically
  ``[B, T, C, D]``).

**Shape inference and type rules**

* ``hc_count`` and ``hidden_size`` must be positive.
* The residual rank must be at least 2, and its last dimension must be compatible with
  ``hc_count * hidden_size``.
* READ and FINAL_MIX require four inputs with floating-point weights, five when ``produce_write_gate`` is true, six (without zero points) or eight (with zero points) for compressed weights, or eight (without) or eleven (with zero points) for compressed weights with a write gate (see the input forms above).  
  The write-gate inputs are always the last inputs.  
  FINAL_MIX must not produce a write gate. READ/FINAL_MIX require ``lowrank > 0``.
* ``norm_weight`` must have shape ``[C*D]``, ``w_down`` ``[L, C*D]``, ``w_up`` ``[C*D, L]``, and
  ``w_inject`` (if present) ``[C, C*D]``. ``w_down``, ``w_up`` and ``w_inject`` (if present) must share one element type.
* For the compressed form, scale tensors must be rank 2 with last dimension equal to the weight's
  output channel count (``L`` for ``w_down``, ``C*D`` for ``w_up``, ``C`` for ``w_inject``), zero points (when present) must have the
  same shape as their scales, scales must be ``f16``, zero points must have the weight element type (``i4``, ``u4``, ``i8`` or ``u8``), and ``K`` must be divisible by the group count ``G``
  (the group size ``K / G`` is derived from the shapes, see the compressed inputs above).
* WRITE requires exactly three inputs and returns the residual shape. ``sublayer_y`` must have
  last dimension ``D``, ``write_gate`` must have last dimension ``C``, and both must share the
  residual element type.
* READ/FINAL_MIX output 0 preserves the residual rank and leading dimensions and replaces the
  final dimension with ``hidden_size``. When the write gate is produced, output 1 preserves the
  leading dimensions and replaces the final dimension with ``hc_count``.
* All outputs use the element type of ``residual``.

**Implementation notes**

The specification places no limit on ``hc_count``, ``hidden_size`` or ``lowrank`` beyond being
positive. A plugin may support only a subset of geometries, element types, quantization group
sizes, or the compressed-weight form, and is expected to reject or not fuse the operation otherwise.


**Example**

The example below shows READ with four branches, per-branch hidden width 64, low-rank width 32,
and a generated write gate. The flattened residual has shape ``[1, 2, 256]``.

.. code-block:: xml
   :force:

   <layer ... type="GatedResidual">
       <data mode="0" hc_count="4" hidden_size="64" lowrank="32"
             epsilon="1e-6" norm_gain_plus_one="true"
             produce_write_gate="true"/>
       <input>
           <port id="0">  <!-- residual: [B,T,C*D] -->
               <dim>1</dim><dim>2</dim><dim>256</dim>
           </port>
           <port id="1">  <!-- norm_weight: [C*D] -->
               <dim>256</dim>
           </port>
           <port id="2">  <!-- w_down: [L,C*D] -->
               <dim>32</dim><dim>256</dim>
           </port>
           <port id="3">  <!-- w_up: [C*D,L] -->
               <dim>256</dim><dim>32</dim>
           </port>
           <port id="4">  <!-- w_inject: [C,C*D] -->
               <dim>4</dim><dim>256</dim>
           </port>
       </input>
       <output>
           <port id="5">  <!-- mixed: [B,T,D] -->
               <dim>1</dim><dim>2</dim><dim>64</dim>
           </port>
           <port id="6">  <!-- write_gate: [B,T,C] -->
               <dim>1</dim><dim>2</dim><dim>4</dim>
           </port>
       </output>
   </layer>

The matching WRITE that adds the sub-layer output back to every branch:

.. code-block:: xml
   :force:

   <layer ... type="GatedResidual">
       <data mode="1" hc_count="4" hidden_size="64" lowrank="32"
             epsilon="1e-6" norm_gain_plus_one="true"
             produce_write_gate="false"/>
       <input>
           <port id="0">  <!-- residual: [B,T,C*D] -->
               <dim>1</dim><dim>2</dim><dim>256</dim>
           </port>
           <port id="1">  <!-- sublayer_y: [B,T,D] -->
               <dim>1</dim><dim>2</dim><dim>64</dim>
           </port>
           <port id="2">  <!-- write_gate: [B,T,C] -->
               <dim>1</dim><dim>2</dim><dim>4</dim>
           </port>
       </input>
       <output>
           <port id="3">  <!-- residual_out: [B,T,C*D] -->
               <dim>1</dim><dim>2</dim><dim>256</dim>
           </port>
       </output>
   </layer>

A FINAL_MIX with ``u4`` compressed weights (eight-input form). ``w_down`` has ``K = C*D = 256``
and ``G_d = 8``, so its group size is ``256 / 8 = 32``; ``w_up`` has ``K = L = 32`` and ``G_u = 1``,
so it is per-channel (group size 32 = K):

.. code-block:: xml
   :force:

   <layer ... type="GatedResidual">
       <data mode="2" hc_count="4" hidden_size="64" lowrank="32"
             epsilon="1e-6" norm_gain_plus_one="true"
             produce_write_gate="false"/>
       <input>
           <port id="0">  <!-- residual: f16 [B,T,C*D] -->
               <dim>1</dim><dim>2</dim><dim>256</dim>
           </port>
           <port id="1">  <!-- norm_weight: f16 [C*D] -->
               <dim>256</dim>
           </port>
           <port id="2">  <!-- w_down: u4 [L,C*D] -->
               <dim>32</dim><dim>256</dim>
           </port>
           <port id="3">  <!-- w_up: u4 [C*D,L] -->
               <dim>256</dim><dim>32</dim>
           </port>
           <port id="4">  <!-- scale_down: f16 [G_d,L] -->
               <dim>8</dim><dim>32</dim>
           </port>
           <port id="5">  <!-- scale_up: f16 [G_u,C*D] -->
               <dim>1</dim><dim>256</dim>
           </port>
           <port id="6">  <!-- zp_down: u4 [G_d,L] -->
               <dim>8</dim><dim>32</dim>
           </port>
           <port id="7">  <!-- zp_up: u4 [G_u,C*D] -->
               <dim>1</dim><dim>256</dim>
           </port>
       </input>
       <output>
           <port id="8">  <!-- mixed: f16 [B,T,D] -->
               <dim>1</dim><dim>2</dim><dim>64</dim>
           </port>
       </output>
   </layer>

**Types**

* *T*: floating-point element type of the data inputs.
* *T_W*: ``i4``, ``u4``, ``i8`` or ``u8`` for compressed ``w_down`` / ``w_up`` / ``w_inject`` (compressed forms only); in the uncompressed forms they use a floating-point type.
* Outputs use the element type of ``residual``.
