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
 * Expected Loop body semantics per step (state size N, head dim P):
 * 1) Squeeze the per-step inputs: `dA_t`, `dBx_t`, `C_t` over the sequence axis.
 * 2) Update recurrent state: `state_t = state_{t-1} * dA_t + dBx_t`
 * 3) Compute per-step output: `y_t = reduce_sum(state_t * unsqueeze(C_t), axis=N)` and scatter to
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
///      ┌────────┐   ┌────────┐   ┌────────┐   ┌───────────────┐
///      │   dA   │   │  dBx   │   │   C    │   │Recurrent State│
///      └───┬────┘   └───┬────┘   └───┬────┘   └──────┬────────┘
///          │            │            │               │
///      ┌───┴────────────┴────────────┴───────────────┴────────┐
///      │                  Loop (recurrent body)               │
///      └──────────────────────────┬───────────────────────────┘
///                                 │
///                  ┌──────────────┴──────────────┐
///                  │ Concat / Slice / Reshape(s) │
///                  └──────────────┬──────────────┘
///                                 │
///                   ┌─────────────┴─────────────┐
///                   │     Output, StateOut      │
///                   └───────────────────────────┘
///
/// After:
///      ┌────────┐   ┌────────┐   ┌────────┐   ┌───────────────┐
///      │   dA   │   │  dBx   │   │   C    │   │Recurrent State│
///      └───┬────┘   └───┬────┘   └───┬────┘   └──────┬────────┘
///          │            │            │               │
///      ┌───┴────────────┴────────────┴───────────────┴────────┐
///      │                       Mamba2                         │
///      └──────────────────────────┬───────────────────────────┘
///                                 │
///                   ┌─────────────┴─────────────┐
///                   │     Output, StateOut      │
///                   └───────────────────────────┘

class TRANSFORMATIONS_API Mamba2Fusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("Mamba2Fusion");
    Mamba2Fusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::pass
