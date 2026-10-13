// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// @brief Folds inverse Transpose operations around an RMS into a feature-axis RMS.
///
/// The pass changes the following graph:
///
///     input -> Transpose(order) -> RMS(axis=-1) -> Transpose(inverse_order)
///
/// to:
///
///     input -> RMS(axis=1)
///
/// The gamma input is preserved. The transformation applies to rank-4 and rank-5
/// inputs when the first Transpose moves feature axis 1 to the last dimension,
/// both Transpose orders are constant and inverse to each other, RMS uses
/// elementwise affine scaling, and intermediate nodes have a single consumer.
class FoldRMSTransposes : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FoldRMSTransposes");
    FoldRMSTransposes();
};

}  // namespace ov::intel_gpu
