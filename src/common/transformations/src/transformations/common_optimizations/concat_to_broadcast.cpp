// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_to_broadcast.hpp"

#include "itt.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConcatToBroadcast::ConcatToBroadcast() {
    MATCHER_SCOPE(ConcatToBroadcast);

    auto concat_label = pattern::wrap_type<op::v0::Concat>([](const Output<Node>& value) {
        auto node = value.get_node_shared_ptr();
        auto first_input_source_output = node->get_input_source_output(0);

        // Do not start comparing from 0 (with itself)
        const auto& input_values = node->input_values();
        auto it = input_values.cbegin();
        std::next(it, 1);

        for (; it != input_values.cend(); ++it) {
            if (*it != first_input_source_output) {
                return false;
            }
        }

        return true;
    });

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto concat = pattern_map.at(concat_label).get_node_shared_ptr();

        if (transformation_callback(concat)) {
            return false;
        }

        if (concat->get_default_output().get_partial_shape().is_dynamic()) {
            return false;
        }

        const auto& broadcast_input = concat->input_value(0);
        auto target_shape = 
            std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                   Shape{concat->get_default_output().get_shape().size()},
                                                   concat->get_default_output().get_shape());
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(broadcast_input, target_shape);
        broadcast->set_friendly_name(concat->get_friendly_name());
        ov::replace_node(concat, broadcast);

        ov::copy_runtime_info(concat, broadcast);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(concat_label, matcher_name);
    this->register_matcher(m, callback);
}