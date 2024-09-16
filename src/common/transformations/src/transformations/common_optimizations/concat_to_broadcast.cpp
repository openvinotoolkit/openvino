// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_to_broadcast.hpp"

#include "itt.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

static bool use_broadcast(const std::shared_ptr<ov::op::v0::Concat>& concat) {
    const auto& output = concat->output(0);
    const auto& input = concat->input(0);
    const auto& input_concat_dim = input.get_partial_shape()[concat->get_axis()];

    return input_concat_dim.is_static() && input_concat_dim.get_length() == 1 && output.get_partial_shape().is_static();
}

ov::pass::ConcatToBroadcast::ConcatToBroadcast() {
    MATCHER_SCOPE(ConcatToBroadcast);

    auto concat_label = pattern::wrap_type<op::v0::Concat>([](const Output<Node>& value) {
        auto node = value.get_node_shared_ptr();
        if (node->output(0).get_partial_shape().rank().is_dynamic()) {
            return false;
        }

        auto first_input_source_output = node->get_input_source_output(0);
        if (first_input_source_output.get_partial_shape().rank().is_dynamic()) {
            return false;
        }

        const auto& input_values = node->input_values();

        return std::all_of(input_values.cbegin(), input_values.cend(), [&](const ov::Output<Node>& output) {
            return first_input_source_output == output;
        });
    });

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto root_node = pattern_map.at(concat_label).get_node_shared_ptr();
        auto concat = ov::as_type_ptr<op::v0::Concat>(root_node);
        if (!concat) {
            return false;
        }

        if (transformation_callback(concat)) {
            return false;
        }

        const auto& input = concat->input_value(0);

        std::shared_ptr<Node> replacement;
        if (use_broadcast(concat)) {
            auto target_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                       Shape{concat->output(0).get_shape().size()},
                                                                       concat->output(0).get_shape());
            replacement = std::make_shared<ov::op::v3::Broadcast>(input, target_shape);
        } else {
            return false;
        }

        /* Common case (converting to Tile) causes an issue in e2e test with unknown root cause (ticket: 142246)
        else {
            std::vector<size_t> repeat_num_vec(concat->output(0).get_partial_shape().rank().get_length(), 1);
            repeat_num_vec[concat->get_concatenation_axis()] = concat->get_input_size();

            auto repeat_num =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{repeat_num_vec.size()}, repeat_num_vec);
            replacement = std::make_shared<ov::op::v0::Tile>(input, repeat_num);
        }
        */

        replacement->set_friendly_name(concat->get_friendly_name());
        ov::replace_node(concat, replacement);

        ov::copy_runtime_info(concat, replacement);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(concat_label, matcher_name);
    this->register_matcher(m, callback);
}
