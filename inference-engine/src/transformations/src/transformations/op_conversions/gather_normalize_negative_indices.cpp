// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/gather_normalize_negative_indices.hpp"

#include <memory>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::GatherNegativeIndicesNormalize, "GatherNegativeIndicesNormalize", 0);

ngraph::pass::GatherNegativeIndicesNormalize::GatherNegativeIndicesNormalize() {
    MATCHER_SCOPE(GatherNegativeIndicesNormalize);
    auto gather = ngraph::pattern::wrap_type<opset7::Gather>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto gather_node = std::dynamic_pointer_cast<ngraph::opset7::Gather>(pattern_to_output.at(gather).get_node_shared_ptr());

        if (gather_node == nullptr) {
            return false;
        }

        auto data_input = gather_node->input_value(0);
        auto indices_input = gather_node->input_value(1);

        if (!data_input.get_partial_shape().rank().is_static()) {
            return false;
        }

        auto indices_rank = indices_input.get_partial_shape().rank();
        if (!indices_rank.is_static() || indices_rank.get_length() != 0) {
            return false;
        }

        auto indices_constant = std::dynamic_pointer_cast<ngraph::opset7::Constant>(indices_input.get_node_shared_ptr());
        if (!indices_constant) {
            return false;
        }

        auto indices = indices_constant->cast_vector<int64_t>()[0];
        if (indices >= 0) {
            return false;
        }

        auto shape_of_node = std::make_shared<ngraph::opset7::ShapeOf>(data_input);
        auto input_gather_node = std::make_shared<ngraph::opset7::Gather>(shape_of_node,
             gather_node->input_value(2), ngraph::opset7::Constant::create(ngraph::element::i32, Shape{1}, {0}));
        auto cast_node = std::make_shared<ngraph::opset7::Convert>(input_gather_node, ngraph::element::i32);
        auto add_node = std::make_shared<ngraph::opset7::Add>(cast_node, indices_input);
        auto gather_new = std::make_shared<ngraph::opset7::Gather>(data_input, add_node, gather_node->input_value(2));
        gather_new->set_friendly_name(gather_node->get_friendly_name());

        ngraph::copy_runtime_info(gather_node, {shape_of_node, input_gather_node, cast_node, add_node, gather_new});
        ngraph::replace_node(gather_node, gather_new);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather, matcher_name);
    register_matcher(m, callback);
}
