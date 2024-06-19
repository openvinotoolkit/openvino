// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sdpa_to_paged_attention/position_ids_replacer.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
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
        const auto& pattern_map = m.get_pattern_value_map();
        replace_node(pattern_map.at(position_ids_pattern).get_node_shared_ptr(), position_ids.get_node_shared_ptr());
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}