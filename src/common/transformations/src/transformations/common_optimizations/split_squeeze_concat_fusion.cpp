// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/split_squeeze_concat_fusion.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::SplitSqueezeConcatFusion::SplitSqueezeConcatFusion() {
    MATCHER_SCOPE(SplitSqueezeConcatFusion);
    // Detect only concat, because we don't know how many inputs will go into concat
    auto concat_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Concat>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto concat =
            std::dynamic_pointer_cast<ov::op::v0::Concat>(pattern_to_output.at(concat_pattern).get_node_shared_ptr());
        if (!concat)
            return false;

        NodeVector nodes_to_delete{concat};

        int64_t axis_value = 0;
        std::shared_ptr<ov::op::v1::Split> split;

        const auto& concat_inputs = concat->input_values();
        if (concat_inputs.empty())
            return false;
        for (size_t i = 0; i < concat_inputs.size(); i++) {
            auto squeeze = std::dynamic_pointer_cast<ov::op::v0::Squeeze>(concat_inputs[i].get_node_shared_ptr());
            if (!squeeze)
                return false;

            nodes_to_delete.push_back(squeeze);

            auto split_to_check = std::dynamic_pointer_cast<ov::op::v1::Split>(squeeze->get_input_node_shared_ptr(0));
            if (!split_to_check)
                return false;
            std::vector<int64_t> squeeze_axes_vec;
            if (squeeze->get_input_size() < 2) {
                const auto& shape = squeeze->get_input_partial_shape(0);
                if (shape.is_dynamic()) {
                    return false;
                }
                for (size_t i = 0; i < shape.size(); i++) {
                    if (shape[i].get_length() == 1)
                        squeeze_axes_vec.push_back(static_cast<int64_t>(i));
                }

            } else {
                auto squeeze_axes =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(squeeze->get_input_node_shared_ptr(1));
                if (!squeeze_axes)
                    return false;
                squeeze_axes_vec = squeeze_axes->cast_vector<int64_t>();
            }

            if (squeeze_axes_vec.size() != 1)
                return false;

            if (i == 0) {
                axis_value = squeeze_axes_vec[0];
                nodes_to_delete.push_back(split_to_check);
                split = split_to_check;
            } else if (axis_value != squeeze_axes_vec[0] || split_to_check != split) {
                return false;
            }

            auto split_output = squeeze->input_value(0);
            if (split_output.get_target_inputs().size() != 1 || split_output.get_index() != i)
                return false;
        }

        if (split->get_num_splits() != concat_inputs.size())
            return false;

        auto split_axis = std::dynamic_pointer_cast<ov::op::v0::Constant>(split->input_value(1).get_node_shared_ptr());
        if (!split_axis)
            return false;

        auto axis_vec = split_axis->cast_vector<int64_t>();
        if (axis_vec.size() != 1 || axis_value != axis_vec[0])
            return false;

        auto input = split->input_value(0);

        auto concat_axis = concat->get_axis();
        auto rank = input.get_partial_shape().rank();
        if (!rank.is_static())
            return false;
        std::vector<int64_t> order(rank.get_length());
        std::iota(order.begin(), order.end(), 0);
        order.erase(order.begin() + axis_value);
        order.insert(order.begin() + concat_axis, axis_value);

        auto transpose_order = ov::op::v0::Constant::create(element::i64, {(size_t)rank.get_length()}, order);
        auto transpose = register_new_node<ov::op::v1::Transpose>(input, transpose_order);
        auto shape_after = ov::op::v0::Constant::create(element::i64,
                                                        {(size_t)rank.get_length() - 1},
                                                        concat->get_output_partial_shape(0).to_shape());
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, shape_after, false);

        reshape->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(nodes_to_delete, {transpose, reshape});
        ov::replace_node(m.get_match_root(), reshape);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_pattern, matcher_name);
    register_matcher(m, callback);
}
