// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

// BGM→MOE passes (GatherMatmul graph → MOE op, used by GPU)
class TRANSFORMATIONS_API Convert2GatherMatmulMoeBlockToMoeOp;
class TRANSFORMATIONS_API Convert3GatherMatmulMoeBlockToMoeOp;
class TRANSFORMATIONS_API MoeOpFusion;

}  // namespace ov::pass

// BGM→MOE passes — convert post-BGM graph (GatherMatmul + compact routing) to MOE op.
// When BGMCompressed nodes are present, produces MOECompressed instead of MOE.
class ov::pass::Convert2GatherMatmulMoeBlockToMoeOp : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Convert2GatherMatmulMoeBlockToMoeOp");
    Convert2GatherMatmulMoeBlockToMoeOp();
};

class ov::pass::Convert3GatherMatmulMoeBlockToMoeOp : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Convert3GatherMatmulMoeBlockToMoeOp");
    // allow_scale_expansion = false leaves GEMMs that would need per-channel scale expansion unfused.
    explicit Convert3GatherMatmulMoeBlockToMoeOp(bool allow_scale_expansion = true);
};

class ov::pass::MoeOpFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("MoeOpFusion");
    explicit MoeOpFusion(bool allow_scale_expansion = true) {
        add_matcher<ov::pass::Convert2GatherMatmulMoeBlockToMoeOp>();
        add_matcher<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>(allow_scale_expansion);
    }
};
