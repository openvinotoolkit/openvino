// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_one_hot_v16_to_v1.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertOneHot16To1::ConvertOneHot16To1() {
    MATCHER_SCOPE(ConvertOneHot16To1);

    auto one_hot_v16 = pattern::wrap_type<ov::op::v16::OneHot>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto one_hot_v16_node = ov::as_type_ptr<ov::op::v16::OneHot>(m.get_match_root());
        if (!one_hot_v16_node)
            return false;

        if (one_hot_v16_node->get_negative_indices_mode() == ov::op::v16::OneHot::NegativeIndicesMode::NORMALIZE)
            return false;

        auto one_hot_v1_node = std::make_shared<ov::op::v1::OneHot>(one_hot_v16_node->input_value(0),
                                                                    one_hot_v16_node->input_value(1),
                                                                    one_hot_v16_node->input_value(2),
                                                                    one_hot_v16_node->input_value(3),
                                                                    one_hot_v16_node->get_axis());

        one_hot_v1_node->set_friendly_name(one_hot_v16_node->get_friendly_name());
        ov::copy_runtime_info(one_hot_v16_node, one_hot_v1_node);
        ov::replace_node(one_hot_v16_node, one_hot_v1_node);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(one_hot_v16, matcher_name);
    register_matcher(m, callback);
}
