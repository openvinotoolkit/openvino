.. {#openvino_docs_ops_internal_GatedResidual}

GatedResidual
=============

.. meta::
  :description: Learn about GatedResidual, an internal operation for branch-wise gated residual READ, WRITE, and FINAL_MIX computations used by hyper-connection transformer models.

**Versioned name**: *GatedResidual*

**Category**: *Internal*

**Short description**:
The *GatedResidual* operation represents the gated residual (hyper-connection) mechanism used by Qwen4-Exp / Qwen3.8-Flash-Next-style models. It handles branch-wise normalization and gated branch merging, the gated residual update, and an optional terminal merge. READ and WRITE are distinct operation modes; the sub-layer output is an explicit input to WRITE.

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
stored normalization gain ``w`` is interpreted as ``1 + w`` when ``norm_gain_plus_one`` is
true; otherwise it is used directly.

For each branch:

.. math::

   \bar{R}_c = \operatorname{RMSNorm}_{D}(R_c, \gamma_c, \epsilon)

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

**Attributes**

* *mode*

  * **Description**: Selects ``READ``, ``WRITE``, or ``FINAL_MIX``.
  * **Type**: ``int`` (enum: ``READ=0``, ``WRITE=1``, ``FINAL_MIX=2``)
  * **Required**: *yes*

* *hc_count*

  * **Description**: Number of residual branches, :math:`C`.
  * **Type**: ``int``
  * **Required**: *yes*

* *hidden_size*

  * **Description**: Hidden width of each branch, :math:`D`.
  * **Type**: ``int``
  * **Required**: *yes*

* *lowrank*

  * **Description**: Low-rank width, :math:`L`, used by READ and FINAL_MIX projections. It is unused by WRITE.
  * **Type**: ``int``
  * **Required**: *yes*

* *epsilon*

  * **Description**: Epsilon used by branch-wise RMS normalization.
  * **Type**: ``float``
  * **Default value**: ``1e-6``
  * **Required**: *no*

* *norm_gain_plus_one*

  * **Description**: If true, interpret the stored normalization weight as ``1 + w``; otherwise use it as the gain directly.
  * **Type**: ``bool``
  * **Default value**: ``true``
  * **Required**: *no*

* *produce_write_gate*

  * **Description**: In READ mode, compute and emit the write gate using ``W_INJECT``. It must be false for FINAL_MIX and is unused in WRITE mode.
  * **Type**: ``bool``
  * **Default value**: ``false``
  * **Required**: *no*

* *output_type*

  * **Description**: Requested element type of the operation outputs. When dynamic, the output type is inferred from the residual input.
  * **Type**: ``element::Type``
  * **Default value**: ``dynamic``
  * **Required**: *no*

**Inputs**

Input indices depend on ``mode``. All dimensions below use ``B`` for batch, ``T`` for flattened
batch-token rows, ``C`` for branches, ``D`` for per-branch hidden width, and ``L`` for low-rank
width. The logical residual shape is ``[B, T, C, D]``; the flattened representation is
``[B, T, C*D]`` (or an equivalent token-major representation).

* **READ / FINAL_MIX input 0**: ``residual``
  A floating-point tensor with logical shape ``[B, T, C, D]`` and physical shape
  ``[B, T, C*D]``. **Required.**

* **READ / FINAL_MIX input 1**: ``norm_weight``
  A 1D tensor of shape ``[C*D]`` containing the branch-wise normalization gains in flattened
  storage order. Each branch uses its own contiguous ``D``-element segment. **Required.**

* **READ / FINAL_MIX input 2**: ``w_down``
  A 2D tensor of shape ``[L, C*D]`` for the low-rank down projection. **Required.**

* **READ / FINAL_MIX input 3**: ``w_up``
  A 2D tensor of shape ``[C*D, L]`` for the up projection that produces read-gate logits. **Required.**

* **READ input 4**: ``w_inject``
  A 2D tensor of shape ``[C, C*D]``. Required when ``produce_write_gate`` is true; not required otherwise.

* **WRITE input 0**: ``residual``
  A floating-point tensor with logical shape ``[B, T, C, D]`` and physical shape
  ``[B, T, C*D]``. **Required.**

* **WRITE input 1**: ``sublayer_y``
  A floating-point tensor of shape ``[B, T, D]`` (or equivalent token-major shape), containing
  the output of the intervening sub-layer. **Required.**

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
* The residual rank must be at least 2. For the documented logical interpretation, the final
  residual dimension represents ``C*D`` and must equal ``hc_count * hidden_size``.
* READ and FINAL_MIX require exactly four inputs, or five when ``produce_write_gate`` is true.
  FINAL_MIX must not produce a write gate. READ/FINAL_MIX require ``lowrank > 0``.
* WRITE requires exactly three inputs and returns the residual shape.
* READ/FINAL_MIX output 0 preserves the residual rank and leading dimensions and replaces the
  final dimension with ``hidden_size``. When the write gate is produced, output 1 preserves the
  leading dimensions and replaces the final dimension with ``hc_count``.
* The WRITE broadcast contract requires ``sublayer_y`` to match the residual leading dimensions
  and have final width ``D``, and ``write_gate`` to match those leading dimensions and have final
  width ``C``. A gate broadcasts as one scalar per branch across that branch's hidden dimension.
* The READ/FINAL_MIX weight dimensions must match the projection shapes specified in Inputs.
* ``output_type`` defaults to the residual input element type; it may be explicitly specified.
  When it is dynamic, output 0 and any optional output 1 use the residual input element type.

**Example**

The example below shows READ with four branches, per-branch hidden width 64, low-rank width 32,
and a generated write gate. The flattened residual has shape ``[1, 2, 256]``.

.. code-block:: xml
   :force:

   <layer ... type="GatedResidual">
       <data mode="0" hc_count="4" hidden_size="64" lowrank="32"
             epsilon="1e-6" norm_gain_plus_one="true"
             produce_write_gate="true" output_type="f16"/>
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

**Types**

* *T*: floating-point element type of the data inputs.
* Outputs use the element type specified by ``output_type``. When ``output_type`` is dynamic,
  outputs use the element type of ``residual``.
