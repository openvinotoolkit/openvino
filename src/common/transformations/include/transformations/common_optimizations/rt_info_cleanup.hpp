// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief RTInfoCleanup erases "disable_const_folding", "dequantization" and "decompression" attributes from
 * all ops in the ov model.
 */
class TRANSFORMATIONS_API RTInfoCleanup : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RTInfoCleanup");
    explicit RTInfoCleanup() {
        using namespace ov::pass;

        MATCHER_SCOPE(RTInfoCleanup);
        auto any_op = std::make_shared<pattern::op::True>();
        ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
            auto root = m.get_match_root();
            enable_constant_folding(root);
            unmark_dequantization_node(root);
            unmark_as_decompression(root);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(any_op, "RTInfoCleanup");
        this->register_matcher(m, callback);
    }
};

}  // namespace ov::pass
