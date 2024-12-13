// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sdpa_to_paged_attention/position_ids_replacer.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

// TODO: Instead of using the following transformation that matches quite a specific place in a model graph in case when
// position_ids parameter is missing, consider replacing always existing attention_mask parameter with a sub-graph using
// a new slot_mapping parameter.
ov::pass::PositionIDsReplacer::PositionIDsReplacer(const Output<Node>& position_ids) {
    MATCHER_SCOPE(PositionIDsReplacer);

    auto input_ids = pattern::any_input();
    auto input_embed = pattern::wrap_type<v8::Gather>({pattern::any_input(), input_ids, pattern::any_input()});

    auto position_ids_pattern = pattern::any_input();
    auto offset = pattern::wrap_type<v0::Constant>();
    auto add_offset = pattern::wrap_type<v1::Add>({position_ids_pattern, offset});
    auto convert = pattern::wrap_type<v0::Convert>({add_offset});
    auto position_embed = pattern::wrap_type<v8::Gather>({pattern::any_input(), convert, pattern::any_input()});

    auto mul = pattern::optional<v0::MatMul>({input_embed, pattern::any_input()});

    auto add = pattern::wrap_type<v1::Add>({mul, position_embed});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        // std::cout << "XXXXXX PositionIDsReplacer" << std::endl;
        const auto& pattern_map = m.get_pattern_value_map();
        replace_node(pattern_map.at(position_ids_pattern).get_node_shared_ptr(), position_ids.get_node_shared_ptr());
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::pass::PositionIDsReplacerQwen::PositionIDsReplacerQwen(const Output<Node>& position_ids) {
    MATCHER_SCOPE(PositionIDsReplacerQwen);

    auto max_context_len_pattern = pattern::wrap_type<v0::Parameter>();
    auto optional_convert = pattern::optional<v0::Convert>(max_context_len_pattern);
    auto optional_reshape = pattern::optional<v1::Reshape>({optional_convert, pattern::any_input()});

    auto slice_1_pattern = pattern::wrap_type<v8::Slice>(
        {pattern::any_input(), pattern::any_input(), optional_reshape, pattern::any_input(), pattern::any_input()});
    auto slice_2_pattern = pattern::wrap_type<v8::Slice>(
        {slice_1_pattern, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto max_context_len = pattern_map.at(max_context_len_pattern).get_node_shared_ptr();
        if (max_context_len->get_friendly_name() != "max_context_len") {
            // std::cout << "XXXX return false;" << std::endl;
            return false;
        }

        auto slice_1 = pattern_map.at(slice_1_pattern).get_node_shared_ptr();
        auto slice_2 = pattern_map.at(slice_2_pattern).get_node_shared_ptr();

        auto gather =
            std::make_shared<v8::Gather>(slice_1, position_ids, v0::Constant::create(element::i64, Shape{}, {1}));
        gather->set_friendly_name(slice_2->get_friendly_name());
        auto axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
        auto squeeze = std::make_shared<v0::Squeeze>(gather, axis);

        auto reshape_shape = v0::Constant::create(element::i64, Shape{4}, {-1, 1, 1, 128});
        auto reshape = std::make_shared<v1::Reshape>(squeeze, reshape_shape, false);
        replace_node(slice_2, reshape);

        gather->validate_and_infer_types();
        /*        std::cout << "slice_2 in(0) " << slice_2->input(0).get_partial_shape() << std::endl;
                std::cout << "slice_2 out " << slice_2->output(0).get_partial_shape() << std::endl;
                std::cout << "gather in " << gather->input(0).get_partial_shape() << std::endl;
                std::cout << "gather out " << gather->output(0).get_partial_shape() << std::endl;*/
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(slice_2_pattern, matcher_name);
    register_matcher(m, callback);
}
