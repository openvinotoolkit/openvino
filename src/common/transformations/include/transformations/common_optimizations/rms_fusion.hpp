// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/backward_graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RMSFusionMatcher;
class TRANSFORMATIONS_API RMSFusion;

}  // namespace pass
}  // namespace ov

class ov::pass::RMSFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RMSFusionMatcher");
    RMSFusionMatcher(bool force_tail_convert = true, bool enable_without_gamma = false);
};

class ov::pass::RMSFusion : public ov::pass::BackwardGraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("RMSFusion");
    RMSFusion(bool force_tail_convert = true, bool enable_without_gamma = false) {
        add_matcher<RMSFusionMatcher>(force_tail_convert, enable_without_gamma);
    }
};
