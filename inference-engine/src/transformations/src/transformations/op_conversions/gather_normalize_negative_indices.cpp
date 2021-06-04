// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/gather_normalize_negative_indices.hpp"

#include <memory>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::GatherNegativeConstIndicesNormalize, "GatherNegativeConstIndicesNormalize", 0);

ngraph::pass::GatherNegativeConstIndicesNormalize::GatherNegativeConstIndicesNormalize() {
    MATCHER_SCOPE(GatherNegativeConstIndicesNormalize);
    auto data_input = ngraph::pattern::any_input();
    auto axis_input = ngraph::pattern::any_input();
    auto const_indices_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto gather_node = std::make_shared<ngraph::opset7::Gather>(data_input, const_indices_input, axis_input);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto gather = std::dynamic_pointer_cast<ngraph::opset7::Gather>(pattern_to_output.at(gather_node).get_node_shared_ptr());
        auto data = pattern_to_output.at(data_input);
        auto axis = pattern_to_output.at(axis_input);
        auto indices_constant = std::dynamic_pointer_cast<ngraph::opset7::Constant>(pattern_to_output.at(const_indices_input).get_node_shared_ptr());

        if (gather == nullptr) {
            return false;
        }

        if (!data.get_partial_shape().rank().is_static()) {
            return false;
        }

        auto indices = indices_constant->cast_vector<int64_t>();
        if (indices.size() != 1 || indices[0] >= 0) {
            return false;
        }

        auto shape_of = std::make_shared<ngraph::opset7::ShapeOf>(data);
        auto input_gather = std::make_shared<ngraph::opset7::Gather>(shape_of,
            axis, ngraph::opset7::Constant::create(ngraph::element::i32, Shape{1}, {0}));
        auto cast = std::make_shared<ngraph::opset7::Convert>(input_gather, ngraph::element::i32);
        auto add = std::make_shared<ngraph::opset7::Add>(cast, indices_constant);
        auto gather_new = std::make_shared<ngraph::opset7::Gather>(data, add, axis);
        gather_new->set_friendly_name(gather->get_friendly_name());

        ngraph::copy_runtime_info(gather, {shape_of, input_gather, cast, add, gather_new});
        ngraph::replace_node(gather, gather_new);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather_node, matcher_name);
    register_matcher(m, callback);
}
