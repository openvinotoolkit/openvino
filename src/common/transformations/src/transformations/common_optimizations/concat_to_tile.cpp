// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_to_tile.hpp"

#include "itt.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

static bool use_broadcast(const std::shared_ptr<ov::Node> node) {
    if (auto concat = std::static_pointer_cast<ov::op::v0::Concat>(node)) {
        return concat->get_default_output().get_partial_shape().is_static() &&
               concat->get_concatenation_axis() == 1;
    }

    return false;
}

ov::pass::ConcatToTile::ConcatToTile() {
    MATCHER_SCOPE(ConcatToTile);

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


        const auto& input = concat->input_value(0);
        auto target_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                   Shape{concat->get_default_output().get_shape().size()},
                                                   concat->get_default_output().get_shape());
        std::shared_ptr<Node> replacement = use_broadcast(concat) ?
                            std::static_pointer_cast<ov::Node>(std::make_shared<ov::op::v3::Broadcast>(input, target_shape)) :
                            std::static_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Tile>(input, target_shape));
        replacement->set_friendly_name(concat->get_friendly_name());
        ov::replace_node(concat, replacement);

        ov::copy_runtime_info(concat, replacement);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(concat_label, matcher_name);
    this->register_matcher(m, callback);
}