// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Removes the flatten/Concat/Slice/Reshape pattern inserted after the SSM Loop and
 *        reconnects the consumers directly to the Loop outputs.
 *
 * The exporter packs the Loop's two results (the main output and the final recurrent state) into a
 * single tensor to satisfy a single-output op interface: each result is flattened to 1D and
 * concatenated; downstream the blob is sliced back into two segments and reshaped to the original
 * shapes. This flatten/Concat/Slice/Reshape round-trip is a semantic identity. This pass detects it
 * and rewires the consumers straight to `Loop` output 0 (output) and output 1 (recurrent state),
 * which both removes the redundant glue and exposes the native two-output Loop that
 * `FuseSSMLoop` then replaces with `SelectiveSSM`.
 *
 * Before:
 *  ┌──────────────┐        ┌──────────────┐
 *  │ Loop output0 │        │ Loop output1 │
 *  │  (output)    │        │   (state)    │
 *  └──────┬───────┘        └──────┬───────┘
 *         │                       │
 *  ┌──────┴───────┐        ┌──────┴───────┐
 *  │ Reshape([-1])│        │ Reshape([-1])│   flatten each result to 1D
 *  └──────┬───────┘        └──────┬───────┘
 *         │                       │
 *         └───────────┬───────────┘
 *                     │
 *              ┌──────┴──────┐
 *              │   Concat    │                pack both into one 1D blob
 *              └──────┬──────┘
 *              ┌──────┴───────┐
 *          ┌───┴───┐      ┌───┴───┐
 *          │ Slice │      │ Slice │           cut the blob back into two segments
 *          └───┬───┘      └───┬───┘
 *          ┌───┴────┐     ┌───┴────┐
 *          │Reshape │     │Reshape │          restore original shapes
 *          └───┬────┘     └───┬────┘
 *              │              │
 *        [?,?,H,P]        [?,H,P,N]
 *          (output)         (state)
 *
 * After (glue removed, consumers wired to Loop outputs directly):
 *  ┌──────────────┐        ┌──────────────┐
 *  │ Loop output0 │        │ Loop output1 │
 *  └──────┬───────┘        └──────┬───────┘
 *         │                       │
 *    [?,?,H,P]               [?,H,P,N]
 *     (output)                 (state)
 */

class TRANSFORMATIONS_API RemoveConcatSliceAfterLoopSSM : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RemoveConcatSliceAfterLoopSSM");
    RemoveConcatSliceAfterLoopSSM();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses a loop-based selective state-space (SSM) recurrence sub-graph into an internal
 *        SelectiveSSM operation.
 *
 * Discretization is performed ahead of the loop and consumed as 5D per-step slices:
 * `dA = reshape(exp(A * dt), [B, T, H, 1, 1])` (Loop input 2), `dBx = unsqueeze(dt*B, -2) *
 * unsqueeze(x, -1)` (Loop input 3) and `C = unsqueeze(B/C-expanded, -2)` (Loop input 4), where the
 * per-head log-decay `A` is a foldable constant. The loop then consumes the discretized `dA`, `dBx`,
 * `C` and the initial `recurrent_state`. Expected body semantics per step (state size N, head dim P):
 * 1) Squeeze the per-step inputs `dA_t`, `dBx_t`, `C_t` over the sequence axis.
 * 2) Update recurrent state: `state_t = state_{t-1} * dA_t + dBx_t`
 * 3) Compute per-step output: `y_t = reduce_sum(state_t * C_t, axis=N)` and scatter to the current
 *    time index.
 *
 * The matcher validates this body shape/operation pattern before replacing the Loop with
 * `SelectiveSSM`.
 */

class TRANSFORMATIONS_API FuseSSMLoop : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseSSMLoop");
    FuseSSMLoop(size_t& fused_count);
};

/// This pass transforms a loop-based SSM sub-graph into a single internal `SelectiveSSM` operation.
///
/// Before:
///  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌───────────────┐
///  │ A  │ │ dt │ │ B  │ │ x  │ │ C  │ │Recurrent State│
///  └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘ └───────┬───────┘
///    │      │      │      │      │            │
///  ┌─┴──────┴──────┴──────┴──────┴────────────┴───────┐
///  │                Loop (recurrent body)             │
///  └────────────────────────┬─────────────────────────┘
///                           │
///            ┌──────────────┴──────────────┐
///            │ Concat / Slice / Reshape(s) │
///            └──────────────┬──────────────┘
///                           │
///             ┌─────────────┴─────────────┐
///             │     Output, StateOut      │
///             └───────────────────────────┘
///
/// After:
///  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌───────────────┐
///  │ A  │ │ dt │ │ B  │ │ x  │ │ C  │ │Recurrent State│
///  └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘ └───────┬───────┘
///    │      │      │      │      │            │
///  ┌─┴──────┴──────┴──────┴──────┴────────────┴───────┐
///  │                   SelectiveSSM                   │
///  └────────────────────────┬─────────────────────────┘
///                           │
///             ┌─────────────┴─────────────┐
///             │     Output, StateOut      │
///             └───────────────────────────┘

class TRANSFORMATIONS_API SelectiveSSMFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SelectiveSSMFusion");
    SelectiveSSMFusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    size_t get_fused_count() const {
        return m_fused_count;
    }

private:
    size_t m_fused_count = 0;
};

}  // namespace ov::pass
