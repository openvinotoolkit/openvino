// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief RTInfoCleanup erases "disable_const_folding" and "dequantization" attributes from
 * all ops in the ov model.
 */
class TRANSFORMATIONS_API RTInfoCleanup : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RTInfoCleanup");
    explicit RTInfoCleanup() {
        MATCHER_SCOPE(RTInfoCleanup);
        auto any_op = std::make_shared<ov::pass::pattern::op::True>();
        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto root = m.get_match_root();
            ov::pass::enable_constant_folding(root);
            unmark_dequantization_node(root);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(any_op, "RTInfoCleanup");
        this->register_matcher(m, callback);
    }
};

}  // namespace ov::pass
