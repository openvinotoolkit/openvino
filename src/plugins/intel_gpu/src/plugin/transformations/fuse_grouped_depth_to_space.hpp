// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Fuses grouped channel repetition followed by anisotropic depth-to-space.
/// The repeat indices may be produced by the subgraph shown below or by its constant-folded equivalent.
///
/// Before:
///     ┌───────┐    ┌───────────┐    ┌──────┐    ┌───────────┐    ┌─────────┐
///     │ Range ├───►│ Unsqueeze ├───►│ Tile ├───►│ Transpose ├───►│ Reshape │
///     └───────┘    └───────────┘    └──────┘    └───────────┘    └────┬────┘
///                                                                     │ indices
///     ┌───────┐                                                  ┌────▼───┐    ┌─────────┐
///     │ Input ├─────────────────────────────────────────────────►│ Gather ├───►│ Reshape │
///     └───────┘                                                  └────────┘    └────┬────┘
///                                                                                   │
///                                                                             ┌─────▼─────┐
///                                                                             │ Transpose │
///                                                                             └─────┬─────┘
///                                                                                   │
///                                                                              ┌────▼────┐
///                                                                              │ Reshape │
///                                                                              └────┬────┘
///                                                                                   │
///                                                                              ┌────▼────┐
///                                                                              │ [Slice] │
///                                                                              └─────────┘
/// After:
///     ┌───────┐    ┌─────────────────────┐
///     │ Input ├───►│ GroupedDepthToSpace │
///     └───────┘    └─────────────────────┘
///
/// Slice is optional and removes the leading expanded temporal elements.
class FuseGroupedDepthToSpace : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseGroupedDepthToSpace");
    FuseGroupedDepthToSpace();
};

}  // namespace ov::intel_gpu
