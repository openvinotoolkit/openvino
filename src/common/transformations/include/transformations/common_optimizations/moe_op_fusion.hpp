// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

// BGM→MOE passes (GatherMatmul graph → MOE op, used by GPU)
class TRANSFORMATIONS_API Convert2GatherMatmulMoeBlockToMoeOp;
class TRANSFORMATIONS_API Convert3GatherMatmulMoeBlockToMoeOp;
class TRANSFORMATIONS_API MoeOpFusion;

}  // namespace pass
}  // namespace ov

// BGM→MOE passes — convert post-BGM graph (GatherMatmul + compact routing) to MOE op.
// When BGMCompressed nodes are present, produces MOECompressed instead of MOE.
class ov::pass::Convert2GatherMatmulMoeBlockToMoeOp : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Convert2GatherMatmulMoeBlockToMoeOp");
    Convert2GatherMatmulMoeBlockToMoeOp(size_t has_batch_dim = 1, ov::element::Type out_type = ov::element::dynamic);
};

class ov::pass::Convert3GatherMatmulMoeBlockToMoeOp : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Convert3GatherMatmulMoeBlockToMoeOp");
    Convert3GatherMatmulMoeBlockToMoeOp(size_t has_batch_dim = 1, ov::element::Type out_type = ov::element::f16);
};

class ov::pass::MoeOpFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("MoeOpFusion");
    MoeOpFusion(size_t has_batch_dim = 1) {
        add_matcher<ov::pass::Convert2GatherMatmulMoeBlockToMoeOp>(has_batch_dim);
        add_matcher<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>(has_batch_dim);
    }
};
