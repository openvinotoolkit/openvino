// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/extract_reshapes_from_mha.hpp"

#include <openvino/opsets/opset1.hpp>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/snippets_isa.hpp"

using namespace ov::pass;

ov::snippets::pass::ExtractReshapesFromMHA::ExtractReshapesFromMHA() {
    MATCHER_SCOPE(ExtractReshapesFromMHA);
    auto input_m = pattern::any_input(pattern::has_static_shape());
    auto reshape_1_m = pattern::wrap_type<opset1::Reshape>({input_m, pattern::wrap_type<opset1::Constant>()});
    auto sparse_input_1_m = pattern::any_input(pattern::has_static_shape());
    auto sparse_input_2_m = pattern::any_input(pattern::has_static_shape());
    auto add_1_m = pattern::wrap_type<opset1::Add>({reshape_1_m, sparse_input_1_m});
    auto add_2_m = pattern::wrap_type<opset1::Add>({add_1_m, sparse_input_2_m});
    auto reshape_2_m = pattern::wrap_type<opset1::Reshape>({add_2_m, pattern::wrap_type<opset1::Constant>()});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::ExtractReshapesFromMHA")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& input = pattern_map.at(input_m);
        const auto& reshape_2 = pattern_map.at(reshape_2_m);

        const auto& input_shape = input.get_shape();
        const auto& output_shape = reshape_2.get_shape();
        if (input_shape != output_shape)
            return false;

        const auto add_1 = pattern_map.at(add_1_m).get_node_shared_ptr();
        const auto add_2 = pattern_map.at(add_2_m).get_node_shared_ptr();
        const auto& bcast_type = add_1->get_autob();
        if (bcast_type != ov::op::AutoBroadcastType::NUMPY || bcast_type != add_2->get_autob())
            return false;

        const auto& sparse_input_1 = pattern_map.at(sparse_input_1_m);
        const auto& sparse_input_2 = pattern_map.at(sparse_input_2_m);
        auto broadcasted_shape = sparse_input_1.get_partial_shape();
        ov::PartialShape::broadcast_merge_into(broadcasted_shape, sparse_input_2.get_partial_shape(), bcast_type);
        if (ov::shape_size(input_shape) != ov::shape_size(broadcasted_shape.to_shape()))
            return false;

        const auto new_add = std::make_shared<ov::opset1::Add>(sparse_input_1, sparse_input_2);
        const auto target_shape = ov::opset1::Constant::create(ov::element::i32, {input_shape.size()}, input_shape);
        const auto reshape = std::make_shared<ov::opset1::Reshape>(new_add, target_shape, true);
        const auto main_add = std::make_shared<ov::opset1::Add>(input, reshape);

        const auto& old_reshape = pattern_map.at(reshape_2_m);
        return ov::replace_output_update_name(old_reshape, main_add);
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_2_m, matcher_name);
    register_matcher(m, callback);
}
