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
    auto concat = ngraph::pattern::wrap_type<ngraph::opset7::Concat>();

    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto concat_node = std::dynamic_pointer_cast<ngraph::opset7::Concat>(pattern_to_output.at(concat).get_node_shared_ptr());
        if (!concat_node)
            return false;
        auto squeeze0 = std::dynamic_pointer_cast<ngraph::opset7::Squeeze>(concat_node->input_value(0).get_node_shared_ptr());
        if (!squeeze0)
            return false;
        auto split_node = std::dynamic_pointer_cast<ngraph::opset7::Split>(squeeze0->input_value(0).get_node_shared_ptr());
        if (!split_node)
            return false;
        auto axis_node = std::dynamic_pointer_cast<ngraph::opset7::Constant>(split_node->input_value(1).get_node_shared_ptr());
        if (!axis_node)
            return false;
        auto axis_vec = axis_node->cast_vector<int64_t>();
        if (axis_vec.size() != 1)
            return false;
        auto axis_value = axis_vec[0];
        ngraph::NodeVector nodes_to_delete{ split_node, concat_node };
        auto concat_inputs = concat_node->inputs();
        auto split_outputs = split_node->outputs();
        if (concat_inputs.size() != split_outputs.size())
            return false;
        for (size_t i = 0; i < concat_inputs.size(); i++) {
            auto squeeze_node = std::dynamic_pointer_cast<ngraph::opset7::Squeeze>(concat_inputs[i].get_source_output().get_node_shared_ptr());
            if (!squeeze_node)
                return false;
            nodes_to_delete.push_back(squeeze_node);
            auto squeeze_axes = std::dynamic_pointer_cast<ngraph::opset7::Constant>(squeeze_node->input_value(1).get_node_shared_ptr());
            if (!squeeze_axes)
                return false;
            auto squeeze_axes_vec = squeeze_axes->cast_vector<int64_t>();
            if (squeeze_axes_vec.size() != 1 || squeeze_axes_vec[0] != axis_value)
                return false;
            auto split_i_output = split_outputs[i].get_target_inputs();
            if (split_i_output.size() != 1 ||
                split_i_output.begin()->get_node()->get_instance_id() != squeeze_node->get_instance_id())
                return false;
        }

        auto input = split_node->input_value(0);

        auto concat_axis = concat_node->get_axis();
        auto inp_p_shape = input.get_partial_shape();
        if (!inp_p_shape.rank().is_static())
            return false;
        size_t rank = input.get_partial_shape().rank().get_length();
        std::vector<int64_t> order(rank);
        std::iota(order.begin(), order.end(), 0);
        order.erase(order.begin() + axis_value);
        order.insert(order.begin() + concat_axis, axis_value);

        auto transpose_order = ngraph::opset7::Constant::create(element::i64, {rank}, order);
        auto transpose = register_new_node<ngraph::opset7::Transpose>(input, transpose_order);
        auto shape_after = ngraph::opset7::Constant::create(element::i64, { rank - 1 }, concat_node->get_output_shape(0));
        auto reshape = std::make_shared<ngraph::opset7::Reshape>(transpose, shape_after, false);

        reshape->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(nodes_to_delete, {transpose, reshape});
        ngraph::replace_node(m.get_match_root(), reshape);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
