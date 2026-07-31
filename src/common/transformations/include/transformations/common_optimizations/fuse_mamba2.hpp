// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Removes the flatten/Concat/Slice/Reshape pattern inserted after the Mamba2 Loop and
 *        reconnects the consumers directly to the Loop outputs.
 *
 * The exporter packs the Loop's two results (the main output and the final recurrent state) into a
 * single tensor to satisfy a single-output op interface: each result is flattened to 1D and
 * concatenated; downstream the blob is sliced back into two segments and reshaped to the original
 * shapes. This flatten/Concat/Slice/Reshape round-trip is a semantic identity. This pass detects it
 * and rewires the consumers straight to `Loop` output 0 (output) and output 1 (recurrent state),
 * which both removes the redundant glue and exposes the native two-output Loop that
 * `FuseMamba2Loop` then replaces with `Mamba2`.
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

class TRANSFORMATIONS_API RemoveConcatSliceAfterLoopMamba2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RemoveConcatSliceAfterLoopMamba2");
    RemoveConcatSliceAfterLoopMamba2();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses a loop-based Mamba2 selective state-space recurrence sub-graph into an internal
 *        Mamba2 operation.
 *
 * The loop consumes the raw, time-major projections `dt`, `B`, `x`, `C` and the initial
 * `recurrent_state`; the per-head log-decay `A` is a constant embedded in the loop body. Expected
 * body semantics per step (state size N, head dim P):
 * 1) Squeeze the per-step inputs `dt_t`, `B_t`, `x_t`, `C_t` over the sequence axis.
 * 2) Discretize: `dA_t = exp(A * dt_t)` and `dBx_t = (dt_t * B_t) outer x_t`.
 * 3) Update recurrent state: `state_t = state_{t-1} * dA_t + dBx_t`
 * 4) Compute per-step output: `y_t = reduce_sum(state_t * unsqueeze(C_t), axis=N)` and scatter to
 *    the current time index.
 *
 * The matcher validates this body shape/operation pattern before replacing the Loop with `Mamba2`.
 */

class TRANSFORMATIONS_API FuseMamba2Loop : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMamba2Loop");
    FuseMamba2Loop();
};

/// This pass transforms a loop-based Mamba2 sub-graph into a single internal `Mamba2` operation.
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
///  │                      Mamba2                      │
///  └────────────────────────┬─────────────────────────┘
///                           │
///             ┌─────────────┴─────────────┐
///             │     Output, StateOut      │
///             └───────────────────────────┘

class TRANSFORMATIONS_API Mamba2Fusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("Mamba2Fusion");
    Mamba2Fusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::pass
