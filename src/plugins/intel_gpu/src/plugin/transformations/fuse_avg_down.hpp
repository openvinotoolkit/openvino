// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// \brief Replaces canonical AvgDown subgraphs with GPU-friendly operations.
///
/// Before:
///
///   ┌─────────┐     ┌────────────────┐     ┌─────────┐     ┌───────────┐     ┌─────────┐     ┌─────────┐     ┌────────────────────┐
///   │  Input  ├────►│ Pad (optional) ├────►│ Reshape ├────►│ Transpose ├────►│ Reshape ├────►│ Reshape ├────►│ ReduceMean(axis=2) │
///   └─────────┘     └────────────────┘     └─────────┘     └───────────┘     └─────────┘     └─────────┘     └────────────────────┘
///
/// After:
///
///   Spatial downsample:
///   ┌─────────┐     ┌─────────────────────────────┐
///   │  Input  ├────►│ AvgPool(kernel=[1, 2, 2])   │
///   └─────────┘     └─────────────────────────────┘
///
///   Temporal/spatial downsample:
///   ┌─────────┐     ┌─────────────────────┐
///   │  Input  ├────►│ GroupedSpaceToDepth │
///   └─────────┘     └─────────────────────┘
///
///   Identity downsample:
///   ┌─────────┐     ┌─────────┐
///   │  Input  ├────►│ Output  │
///   └─────────┘     └─────────┘
///
/// The optional Pad adds temporal zeros at the beginning and is folded into the replacement.
class FuseAvgDown : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseAvgDown");
    FuseAvgDown();
};

}  // namespace ov::intel_gpu
