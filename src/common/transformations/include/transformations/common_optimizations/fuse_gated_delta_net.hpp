// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GatedDeltaNetFusion;

}  // namespace pass
}  // namespace ov

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
/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses a loop-based gated delta net sub-graph into an internal GatedDeltaNet operation.
 */
class ov::pass::GatedDeltaNetFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatedDeltaNetFusion");
    GatedDeltaNetFusion();
};