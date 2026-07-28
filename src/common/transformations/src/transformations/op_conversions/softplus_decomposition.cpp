// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/softplus_decomposition.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::Matcher;

namespace v0 = ov::op::v0;
ov::pass::SoftPlusDecomposition::SoftPlusDecomposition() {
    MATCHER_SCOPE(SoftPlusDecomposition);
    // decomposes SoftPlus(x) operation into ln(exp(x) + 1.0)
    auto input = ov::pass::pattern::any_input();
    auto softplus = std::make_shared<ov::op::v4::SoftPlus>(input);

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto softplus_input = pattern_to_output.at(input);
        auto softplus_node = pattern_to_output.at(softplus).get_node_shared_ptr();

        if (transformation_callback(softplus_node)) {
            return false;
        }

        auto exp = std::make_shared<v0::Exp>(softplus_input);
        auto add = std::make_shared<ov::op::v1::Add>(
            exp,
            v0::Constant::create(softplus_input.get_element_type(), ov::Shape{1}, {1.0}));
        auto log = std::make_shared<v0::Log>(add);

        log->set_friendly_name(softplus_node->get_friendly_name());
        ov::copy_runtime_info(softplus_node, {exp, add, log});
        ov::replace_node(softplus_node, log);
        return true;
    };

    auto m = std::make_shared<Matcher>(softplus, matcher_name);
    register_matcher(m, callback);
}
