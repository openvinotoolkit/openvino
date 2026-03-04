// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations/common_optimizations/moe_op_fusion.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

// BGM-producing passes (IR → GatherMatmul)
class TRANSFORMATIONS_API ConvertTiledMoeBlockTo2GatherMatmuls;
class TRANSFORMATIONS_API ConvertTiledMoeBlockTo3GatherMatmuls;

class TRANSFORMATIONS_API ConvertTiledMoeBlockToGatherMatmuls;

}  // namespace pass
}  // namespace ov

// BGM-producing passes — create GatherMatmul ops + routing reconstruction
class ov::pass::ConvertTiledMoeBlockTo2GatherMatmuls : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertTiledMoeBlockTo2GatherMatmuls");
    ConvertTiledMoeBlockTo2GatherMatmuls();
};

class ov::pass::ConvertTiledMoeBlockTo3GatherMatmuls : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertTiledMoeBlockTo3GatherMatmuls");
    ConvertTiledMoeBlockTo3GatherMatmuls();
};

// CPU uses BGM-producing passes only (stops at BGMs)
class ov::pass::ConvertTiledMoeBlockToGatherMatmuls : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertTiledMoeBlockToGatherMatmuls");
    ConvertTiledMoeBlockToGatherMatmuls() {
        add_matcher<ov::pass::ConvertTiledMoeBlockTo2GatherMatmuls>();
        add_matcher<ov::pass::ConvertTiledMoeBlockTo3GatherMatmuls>();
    }
};
