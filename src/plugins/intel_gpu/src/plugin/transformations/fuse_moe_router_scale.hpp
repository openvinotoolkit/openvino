// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Fuse a scale on routing weights path to MOECompressed operation
/// by folding the per-expert scale into w2_scale. Scalar and per-expert scales are supported
class FuseMoERouterScale : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoERouterScale");
    FuseMoERouterScale();
};

}  // namespace ov::intel_gpu
