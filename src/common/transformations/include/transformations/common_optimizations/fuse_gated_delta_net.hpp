// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Remove Concat of Loop
 */

class TRANSFORMATIONS_API RemoveConcatSliceAfterLoop : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RemoveConcatSliceAfterLoop");
    RemoveConcatSliceAfterLoop();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses a loop-based gated delta net sub-graph into an internal GatedDeltaNet operation.
 */

class TRANSFORMATIONS_API FuseGDNLoop : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseGDNLoop");
    FuseGDNLoop();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuse l2_norm into GatedDeltaNet
 */

class TRANSFORMATIONS_API FuseL2NormIntoGDN : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseL2NormIntoGDN");
    FuseL2NormIntoGDN();
};

/// This pass transforms a loop-based Gated Delta Net sub-graph to a single internal `GatedDeltaNet` operation.
///
/// Before:
///      ┌───────┐   ┌───────┐   ┌───────┐   ┌──────────────┐   ┌───────┐   ┌────────┐
///      │   Q   │   │   K   │   │   V   │   │Initial State │   │   G   │   │  Beta  │
///      └───┬───┘   └───┬───┘   └───┬───┘   └──────┬───────┘   └───┬───┘   └───┬────┘
///          │           │           │              │               │           │
///          │           │           │              │               │           │
///      ┌───┴───────┐ ┌─┴───────┐ ┌─┴───────┐     │         ┌─────┴─────┐ ┌───┴─────┐
///      │L2Norm(Q)+ │ │L2Norm(K)│ │Transpose│     │         │ Transpose │ │Transpose│
///      │QScale(1/√d)│ └────┬────┘ └────┬────┘     │         └─────┬─────┘ └────┬────┘
///      └─────┬──────┘      │           │          │               │             │
///            │             │           │          │               │             │
///      ┌─────┴─────────────┴───────────┴──────────┴───────────────┴─────────────┴─────┐
///      │                                  Loop (recurrent body)                         │
///      └───────────────────────────────┬─────────────────────────────────────────────────┘
///                                      │
///                       ┌──────────────┴──────────────┐
///                       │ Concat / Slice / Reshape(s) │
///                       └──────────────┬──────────────┘
///                                      │
///                          ┌───────────┴───────────┐
///                          │   CoreAttn, StateOut  │
///                          └───────────────────────┘
///
/// After:
///      ┌───────┐   ┌───────┐   ┌───────┐   ┌──────────────┐   ┌───────┐   ┌────────┐
///      │   Q   │   │   K   │   │   V   │   │Initial State │   │   G   │   │  Beta  │
///      └───┬───┘   └───┬───┘   └───┬───┘   └──────┬───────┘   └───┬───┘   └───┬────┘
///          │           │           │              │               │           │
///          │           │           │              │               │           │
///      ┌───┴───────────┴───────────┴──────────────┴───────────────┴───────────┴──────┐
///      │                              GatedDeltaNet                                    │
///      └───────────────────────────────────┬────────────────────────────────────────────┘
///                                          │
///                               ┌──────────┴──────────┐
///                               │  CoreAttn, StateOut │
///                               └─────────────────────┘

class TRANSFORMATIONS_API GatedDeltaNetFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GatedDeltaNetFusion");
    GatedDeltaNetFusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::pass