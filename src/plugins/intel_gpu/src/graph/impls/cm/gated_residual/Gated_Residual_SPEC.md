# Gated Residual Specification for OpenVINO

**Status:** Design-for-review / Intel GPU plugin implementation in progress

**Audience:** OpenVINO operator and plugin maintainers, graph-transformation reviewers, kernel and performance reviewers

**Scope:** The `GatedResidual` operation, its READ/FINAL_MIX and WRITE modes, graph-fusion passes, Intel GPU implementation, and extensibility requirements.

## 1. Executive summary

Gated Residual (GR), also known as a hyper-connection residual mechanism, widens the residual stream into multiple branches and learns how to merge and update those branches around a transformer sub-layer. It is used by Qwen4-Exp / Qwen3.8-Flash-Next style models.

The proposed OpenVINO operation replaces large, repeatedly decomposed graph regions with one semantically explicit operation:

```text
GatedResidual READ / FINAL_MIX -> sub-layer -> GatedResidual WRITE
```

The operation preserves the model's branch-wise normalization, low-rank gating, scaling factors, and gate ranges. It does not merge unrelated projections or change the semantics of the transformer sub-layer between READ and WRITE.

The Intel GPU plugin provides a `gpu_opset` wrapper and a CM implementation. The implementation selects either a compact fused kernel or a multi-stage DPAS pipeline depending on device and tensor geometry. The generic graph remains available as a fallback whenever matching or implementation constraints are not met.

## 2. Goals and non-goals

### 2.1 Goals

1. Represent the gated-residual semantics explicitly in the OpenVINO graph.
2. Support READ, FINAL_MIX, and WRITE modes with stable input/output contracts.
3. Preserve branch-wise RMS normalization and all model-defined scaling factors.
4. Fuse decomposed graph patterns only when shape, axis, activation, and consumer proofs succeed.
5. Use Intel GPU CM/DPAS execution where the target device and layouts support it.
6. Preserve generic OpenVINO fallback behavior on unsupported devices and shapes.
7. Make mode and attributes serializable, cloneable, and inspectable.
8. Provide a clear path for additional branch counts, dynamic shapes, and future kernel variants.

### 2.2 Non-goals for version 1

- Treating the operation as a generic residual-add replacement.
- Replacing a normal RMSNorm, arbitrary MoE gate, or ordinary residual connection.
- Flattening the branch dimension into one normalization domain.
- Removing the required $1/C$ factors because they appear algebraically redundant.
- Fusing across an intervening sub-layer between READ and WRITE.
- Requiring all devices to implement the fused GPU operation.
- Supporting arbitrary branch counts in the first CM implementation; the current optimized path targets `C = 4`.

## 3. Semantic definition

Let:

- $C$ be the number of residual branches (`hc_count`);
- $D$ be the hidden size of one branch (`hidden_size`);
- $L$ be the low-rank gate width (`lowrank`);
- $R_c \in \mathbb{R}^{D}$ be branch $c$;
- $R \in \mathbb{R}^{C\times D}$ be the widened residual;
- $W_{down} \in \mathbb{R}^{L\times CD}$;
- $W_{up} \in \mathbb{R}^{CD\times L}$;
- $W_{inject} \in \mathbb{R}^{C\times CD}$.

The physical graph representation may use `[B,T,C*D]`; the logical definition is `[B,T,C,D]`.

### 3.1 Branch-wise normalization

For each branch, normalization is performed over its own $D$ channels:

$$
\bar{R}_c = \mathrm{RMSNorm}_{D}(R_c, \gamma_c, \epsilon).
$$

This is not RMS normalization over the flattened $C D$ channels. If `norm_gain_plus_one` is enabled, the effective gain is:

$$
\gamma_c^{effective} = 1 + w_c,
$$

where the checkpoint stores $w_c$.

### 3.2 READ mode

The low-rank read gate is:

$$
T = \mathrm{SiLU}\left(\frac{\bar{R} W_{down}^{T}}{C}\right),
$$

$$
G = \mathrm{sigmoid}(T W_{up}^{T}) \odot \bar{R}.
$$

The merged stream is:

$$
M = \frac{1}{C}\sum_{c=0}^{C-1}G_c.
$$

When `produce_write_gate` is true, the same READ operation also computes the per-branch write gate:

$$
S = 2\cdot \mathrm{sigmoid}\left(\frac{\bar{R}W_{inject}^{T}}{C}\right),
$$

where $S$ has logical shape `[B,T,C]` and range $(0,2)$.

The factors $1/C$ in the low-rank down-projection, merge, and write-gate projection are part of the model definition and must not be folded away unless the transformation proves mathematically equivalent constants and preserves numerical behavior.

### 3.3 WRITE mode

Given sub-layer output $Y \in \mathbb{R}^{B\times T\times D}$ and write gate $S \in \mathbb{R}^{B\times T\times C}$:

$$
R'_c = R_c + S_c \odot Y.
$$

The gate broadcasts over the branch hidden dimension only. Broadcasting over channels or using a gate of shape `[B,T,D]` is a different operation and must not match this fusion.

### 3.4 FINAL_MIX mode

`FINAL_MIX` has READ semantics but does not produce a write gate. It is intended for the model-tail branch mixer or an equivalent terminal use.

## 4. Operation contract

The public/internal base operation is `ov::op::internal::GatedResidual`; the Intel GPU implementation is `ov::intel_gpu::op::GatedResidual` under `gpu_opset`.

### 4.1 Modes

| Mode | Meaning | Inputs | Outputs |
|---|---|---|---|
| `READ` | Normalize, compute low-rank read gate, merge branches, optionally produce write gate | `RESIDUAL`, `NORM_WEIGHT`, `W_DOWN`, `W_UP`, and optional `W_INJECT` | merged stream; optional write gate |
| `WRITE` | Apply a previously computed per-branch write gate to a sub-layer result | `RESIDUAL`, `SUBLAYER_Y`, `WRITE_GATE` | widened residual |
| `FINAL_MIX` | READ-style terminal merge without write-gate generation | `RESIDUAL`, `NORM_WEIGHT`, `W_DOWN`, `W_UP` | merged stream |

### 4.2 READ / FINAL_MIX inputs

| Index | Name | Required shape/meaning |
---:|---|---|
| 0 | `RESIDUAL` | `[B,T,C*D]`; logically `[B,T,C,D]` |
| 1 | `NORM_WEIGHT` | `[C*D]`; branch-wise gains in the flattened storage representation |
| 2 | `W_DOWN` | `[L,C*D]` |
| 3 | `W_UP` | `[C*D,L]` |
| 4 | `W_INJECT` | `[C,C*D]`; required only when `produce_write_gate` is true |

For `FINAL_MIX`, `produce_write_gate` must be false and input 4 is not required.

### 4.3 WRITE inputs

| Index | Name | Required shape/meaning |
|---:|---|---|
| 0 | `RESIDUAL` | `[B,T,C*D]` |
| 1 | `SUBLAYER_Y` | `[B,T,D]` |
| 2 | `WRITE_GATE` | `[B,T,C]`; broadcasts over the branch hidden dimension |

### 4.4 Outputs

- READ/FINAL_MIX output 0: merged stream `[B,T,D]` or the equivalent token-major representation.
- READ output 1, when `produce_write_gate` is true: write gate `[B,T,C]`.
- WRITE output 0: updated widened residual `[B,T,C*D]`.

The operation must preserve the element type selected by the graph/plugin contract. The current optimized CM path requires FP16 layouts; other types must use a supported fallback or remain unfused.

## 5. Attributes and validation

| Attribute | Meaning | Version-1 requirement |
|---|---|---|
| `mode` | `READ`, `WRITE`, or `FINAL_MIX` | Must agree with input count and output count |
| `hc_count` | Branch count $C$ | Optimized CM path currently requires `4` |
| `hidden_size` | Branch width $D$ | Non-zero and compatible with kernel tile requirements |
| `lowrank` | Gate bottleneck width $L$ | Required for READ/FINAL_MIX; bounded by implementation limits |
| `epsilon` | RMSNorm epsilon | Positive finite value |
| `norm_gain_plus_one` | Interpret stored norm gain as `1+w` | Serialized and applied in all READ paths |
| `produce_write_gate` | Emit output 1 in READ | Must be false for FINAL_MIX and consistent with `W_INJECT` |
| `output_type` | Requested operation output type | Must be compatible with input/plugin implementation |

The operation's validation must check:

- input count for the selected mode;
- static or compatible `C` and `D` dimensions where required;
- `W_DOWN`, `W_UP`, and `W_INJECT` geometry;
- write-gate shape `[B,T,C]` and correct broadcast axis;
- valid epsilon and non-zero low-rank width for READ/FINAL_MIX;
- consistent output type and shape.

Validation should produce an OpenVINO exception for an explicitly constructed invalid operation. A graph-fusion pass should prefer a non-match and preserve the generic graph when the source topology cannot be proven.

## 6. Graph-fusion design

### 6.1 READ / FINAL_MIX fusion

`GatedResidualReadFusion` recognizes the decomposed branch-merge graph:

```text
RESIDUAL
  -> Reshape [T,C,D]
  -> branch-wise RMSNorm(width D)
  -> Reshape [T,C*D] --------------------+
                                          |
  -> W_DOWN, scale 1/C, SiLU, W_UP ------> Sigmoid gate * normalized residual
                                          |
  -> Reshape [T,C,D] -> ReduceSum(axis=C) -> scale 1/C -> MIXED
```

When a write gate is requested, the pass additionally proves:

```text
normalized residual -> W_INJECT -> scale 1/C -> Sigmoid -> scale 2 -> WRITE_GATE
```

The pass must prove that the reduction axis is the branch axis, the normalization width is exactly $D$, the branch count is supported, and all required scaling factors are present. It must not match a channel-wise gate or a single RMSNorm over `[C*D]`.

### 6.2 WRITE fusion

`GatedResidualWriteFusion` recognizes:

```text
RESIDUAL [T,C*D]
 + Reshape(WRITE_GATE, [T,C,1]) * Reshape(SUBLAYER_Y, [T,1,D])
 -> Reshape [T,C*D]
```

The pass accepts equivalent `Unsqueeze`-based broadcast forms when axis proof is available. It refuses a gate broadcast along the hidden/channel axis and leaves the original graph intact.

### 6.3 Fusion safety

The passes must preserve observable consumers. If a decomposed intermediate has an additional consumer whose semantics would be changed by replacement, fusion must be rejected or the replacement must preserve that value explicitly. The passes should use structural and shape facts rather than checkpoint-specific tensor names.

## 7. Intel GPU implementation

### 7.1 Implementation selection

The GPU implementation is a `cldnn::gated_residual` primitive with a CM implementation manager. It rejects fused execution when:

- CM JIT or required DPAS support is unavailable;
- an unsupported device architecture is selected;
- the node already contains unsupported fused primitives;
- layouts or element types are not supported;
- branch count, hidden size, low-rank width, or total widened width exceed implementation limits.

The generic graph remains eligible for other plugin implementations.

### 7.2 Compact fused READ kernel

`gr_read.cm` implements the complete READ/FINAL_MIX operation in one kernel when the DPAS pipeline is not selected. Its compile-time parameters include:

- `GR_BRANCHES`;
- `GR_HIDDEN`;
- `GR_LOWRANK`;
- `GR_EPS`;
- `GR_NORM_GAIN_PLUS_ONE`;
- `GR_PRODUCE_WRITE_GATE`.

### 7.3 DPAS READ pipeline

On supported Xe2-class devices and compatible geometry, READ/FINAL_MIX is split into stages so matrix projections use systolic DPAS:

1. `gr_rbar_dpas`: branch-wise RMSNorm and gain into `RBAR` scratch;
2. `gr_down_proj_dpas`: low-rank down projection, $1/C$ scaling, and SiLU into `T_SCRATCH`;
3. `gr_up_proj_dpas`: up projection and read-gate logits into `GATE_SCRATCH`;
4. optional `gr_writegate_dpas`: write-gate projection into output 1;
5. `gr_combine`: sigmoid gating, branch reduction, and merged output.

Stage events are chained in producer-consumer order. Intermediate buffers are explicitly sized for `RBAR`, low-rank scratch, and gate scratch.

### 7.4 WRITE kernel

`gr_write.cm` performs the branch update. It uses a width tile selected from 64, 32, or 16 elements when the hidden width is divisible by that tile, while preserving the exact broadcast semantics.

## 8. Performance design

### 8.1 Fusion benefit

The graph pass removes multiple reshapes, normalization, low-rank projections, activations, broadcasts, reductions, and elementwise operations from the runtime graph. This reduces intermediate tensors and kernel-launch overhead while enabling reuse of data in GPU-local execution stages.

### 8.2 DPAS benefit

The down and up projections are GEMM-like operations. On supported hardware, DPAS stages improve arithmetic throughput and allow projection work to use the systolic path while the surrounding normalization and combine operations remain explicit stages.

### 8.3 Memory behavior

The DPAS path uses three internal FP16 buffers:

- `RBAR`: approximately `num_rows * C * D` elements;
- `T_SCRATCH`: approximately `num_rows * L` elements;
- `GATE_SCRATCH`: approximately `num_rows * C * D` elements.

The compact kernel may avoid these persistent intermediate buffers but can have lower projection throughput. Selection is therefore geometry- and device-dependent.

### 8.4 Dispatch

The operation treats the flattened input as rows of width `C*D`. The number of rows is computed from the input element count and widened width. READ and WRITE use independent dispatch formulas, and the DPAS pipeline chains events explicitly to make scratch-buffer dependencies safe.

## 9. OpenVINO conformance and integration requirements

The implementation follows OpenVINO operation and plugin conventions:

- the internal operation is distinct from the Intel GPU `gpu_opset` wrapper;
- mode, dimensions, epsilon, and boolean attributes are serialized with `visit_attributes()`;
- `validate_and_infer_types()` is the single source of output-shape/type inference;
- `clone_with_new_inputs()` preserves the concrete GPU operation type;
- the operation has stable named input-index structures rather than positional literals in callers;
- matcher passes have independent RTTI names and do not alter graphs on failed proof;
- plugin lowering is registered through the normal ProgramBuilder factory mechanism;
- unsupported devices can use generic decomposed operations or another backend;
- no hidden mutable state is introduced by a stateless GatedResidual node;
- the operation does not take ownership of the sub-layer between READ and WRITE.

The core declaration is marked under development until operation naming, serialization, reference behavior, cross-plugin expectations, and shape/dynamic-rank policy are approved. The optimized implementation is currently Intel GPU-specific.

## 10. Extensibility

### 10.1 Branch counts

The operation contract already carries `hc_count`, so future implementations can support `C != 4`. The matcher must retain branch-axis proofs, and kernel dispatch must provide suitable tiling and write-gate packing for odd and larger branch counts.

### 10.2 Dynamic shapes

A future implementation may support dynamic token count by deferring row-count computation to runtime. Hidden size, branch count, and low-rank widths may still need compile-time constraints for CM JIT specialization. Dynamic shape support must not weaken broadcast or normalization-axis validation.

### 10.3 Data types

FP16 is the current optimized CM format. BF16, FP32, and quantized weights may be added independently, with explicit accumulation and accuracy requirements. Any mixed-precision mode must preserve the gate range and RMSNorm numerical contract.

### 10.4 Kernel variants

Possible future variants include:

- fused READ kernels for additional device generations;
- DPAS kernels with wider tiles or different low-rank blocking;
- fused READ+WRITE only when no observable sub-layer lies between them;
- weight prepacking and cache-friendly layout transformations;
- persistent or cooperative scheduling for long token batches.

### 10.5 Graph-pattern evolution

The matchers should accept semantically equivalent `Reshape`/`Unsqueeze` and commutative elementwise forms only when axes and constants can be proven. Pattern expansion must be accompanied by negative tests to prevent accidental fusion of ordinary residual or channel-gated graphs.

## 11. Verification plan and current evidence

Required tests include:

- mode-specific input-count and shape validation;
- output shape/type inference;
- clone and attribute round trips;
- exact READ numerical comparison against a decomposed reference;
- exact WRITE numerical comparison, including gate broadcast axes;
- FINAL_MIX behavior without write-gate output;
- `norm_gain_plus_one` and epsilon coverage;
- missing or misplaced $1/C$ scale rejection;
- wrong normalization width rejection;
- wrong reduce-axis rejection;
- wrong write-gate broadcast-axis rejection;
- unsupported branch-count and layout fallback;
- READ fusion, WRITE fusion, and decomposed export forms;
- no-consumer-change regression checks;
- GPU compact-kernel and DPAS-pipeline comparisons;
- performance measurements for launch count, intermediate memory, latency, and throughput.

The repository contains host-side graph-fusion tests for direct and decomposed forms, negative tests for unsupported axes and branch structures, and Intel GPU implementation coverage. The committee should treat additional data-type, dynamic-shape, branch-count, serialization, and cross-device coverage as release gates for broader exposure.

## 12. Review decisions requested

The committee is requested to review and approve:

1. the three-mode semantic contract;
2. branch-wise normalization and mandatory scaling-factor semantics;
3. the separation between graph fusion and device-specific lowering;
4. the conservative fallback policy;
5. the current `C = 4`, FP16, CM/DPAS optimization envelope;
6. the plugin-internal status of the operation;
7. the extensibility and validation requirements before broader API promotion.
