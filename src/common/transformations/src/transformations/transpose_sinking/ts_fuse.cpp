// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_fuse.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

TSFuse::TSFuse() {
    MATCHER_SCOPE(TransposeFuse);
    auto transpose_1_label =
        pattern::wrap_type<ov::op::v1::Transpose>({pattern::any_input(), pattern::wrap_type<ov::op::v0::Constant>()},
                                                  CheckTransposeConsumers);
    auto transpose_2_label =
        pattern::wrap_type<ov::op::v1::Transpose>({transpose_1_label, pattern::wrap_type<ov::op::v0::Constant>()});
    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose1 = pattern_to_output.at(transpose_1_label);
        auto transpose2 = pattern_to_output.at(transpose_2_label);
        auto input = transpose1->input_value(0);

        auto transpose1_order = ov::as_type_ptr<ov::op::v0::Constant>(transpose1->get_input_node_shared_ptr(1));
        auto transpose2_order = ov::as_type_ptr<ov::op::v0::Constant>(transpose2->get_input_node_shared_ptr(1));
        if (!transpose1_order || !transpose2_order)
            return false;

        auto order1 = transpose1_order->cast_vector<int64_t>();
        auto order2 = transpose2_order->cast_vector<int64_t>();
        if (order1.size() != order2.size())
            return false;

        bool is_ordered = true;
        for (size_t i = 0; i < order1.size(); i++) {
            if (static_cast<int64_t>(order1.size()) <= order2[i]) {
                return false;
            }
            order2[i] = order1[order2[i]];
            if (order2[i] != static_cast<int64_t>(i))
                is_ordered = false;
        }

        auto transpose_order_type = transpose1_order->get_element_type();
        if (transpose_order_type != transpose2_order->get_element_type())
            transpose_order_type = element::i64;

        if (is_ordered) {
            for (const auto& out_transpose : transpose1->output(0).get_target_inputs()) {
                ov::replace_output_update_name(out_transpose.get_node()->output(0), input);
            }
        } else {
            auto new_order = ov::op::v0::Constant::create(transpose_order_type, {order2.size()}, order2);
            auto new_transpose = register_new_node<ov::op::v1::Transpose>(input, new_order);

            new_transpose->set_friendly_name(m.get_match_root()->get_friendly_name());
            RemoveTransposeConsumers(transpose1);
            copy_runtime_info(transpose1, new_transpose);
            ov::replace_node(transpose1, new_transpose);

            mark_as_no_sinking_node(new_transpose);
        }
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_2_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
