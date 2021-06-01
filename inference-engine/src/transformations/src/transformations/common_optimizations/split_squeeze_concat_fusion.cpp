// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/split_squeeze_concat_fusion.hpp"

#include <memory>
#include <vector>
#include <numeric>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::SplitSqueezeConcatFusion, "SplitSqueezeConcatFusion", 0);

ngraph::pass::SplitSqueezeConcatFusion::SplitSqueezeConcatFusion() {
    MATCHER_SCOPE(SplitSqueezeConcatFusion);
    // Detect only concat, because we don't know how many inputs will go into concat
    auto concat_pattern = ngraph::pattern::wrap_type<ngraph::opset7::Concat>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto concat = std::dynamic_pointer_cast<ngraph::opset7::Concat>(pattern_to_output.at(concat_pattern).get_node_shared_ptr());
        if (!concat) return false;

        NodeVector nodes_to_delete{ concat };

        int64_t axis_value = 0;
        std::shared_ptr<ngraph::opset7::Split> split;

        const auto& concat_inputs = concat->input_values();
        if (concat_inputs.empty()) return false;
        for (size_t i = 0; i < concat_inputs.size(); i++) {
            auto squeeze = std::dynamic_pointer_cast<ngraph::opset7::Squeeze>(concat_inputs[i].get_node_shared_ptr());
            if (!squeeze) return false;

            nodes_to_delete.push_back(squeeze);

            auto split_to_check = std::dynamic_pointer_cast<ngraph::opset7::Split>(squeeze->input_value(0).get_node_shared_ptr());
            auto squeeze_axes = std::dynamic_pointer_cast<ngraph::opset7::Constant>(squeeze->input_value(1).get_node_shared_ptr());
            if (!squeeze_axes || !split_to_check) return false;

            auto squeeze_axes_vec = squeeze_axes->cast_vector<int64_t>();
            if (squeeze_axes_vec.size() != 1) return false;

            if (i == 0) {
                axis_value = squeeze_axes_vec[0];
                nodes_to_delete.push_back(split_to_check);
                split = split_to_check;
            } else if (axis_value != squeeze_axes_vec[0] || split_to_check != split) {
                return false;
            }

            auto split_output = squeeze->input_value(0);
            if (split_output.get_target_inputs().size() != 1 ||
                split_output.get_index() != i)
                return false;
        }

        if (split->get_num_splits() != concat_inputs.size()) return false;

        auto split_axis = std::dynamic_pointer_cast<ngraph::opset7::Constant>(split->input_value(1).get_node_shared_ptr());
        if (!split_axis) return false;

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

        auto transpose_order = ngraph::opset7::Constant::create(element::i64, { (size_t)rank.get_length() }, order);
        auto transpose = register_new_node<ngraph::opset7::Transpose>(input, transpose_order);
        auto shape_after = ngraph::opset7::Constant::create(element::i64, { (size_t)rank.get_length() - 1 }, concat->get_output_shape(0));
        auto reshape = std::make_shared<ngraph::opset7::Reshape>(transpose, shape_after, false);

        reshape->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(nodes_to_delete, { transpose, reshape });
        ngraph::replace_node(m.get_match_root(), reshape);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_pattern, matcher_name);
    register_matcher(m, callback);
}
