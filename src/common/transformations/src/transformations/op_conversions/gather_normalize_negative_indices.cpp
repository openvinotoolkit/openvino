// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/gather_normalize_negative_indices.hpp"

#include <memory>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"

ngraph::pass::GatherNegativeConstIndicesNormalize::GatherNegativeConstIndicesNormalize() {
    MATCHER_SCOPE(GatherNegativeConstIndicesNormalize);
    auto data_input = ngraph::pattern::any_input(pattern::has_static_rank());
    auto axis_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto indices_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto gather_node = ngraph::pattern::wrap_type<op::util::GatherBase>({data_input, indices_input, axis_input});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto gather = pattern_to_output.at(gather_node).get_node_shared_ptr();
        auto data = pattern_to_output.at(data_input);
        auto axis_constant =
            std::dynamic_pointer_cast<ngraph::opset7::Constant>(pattern_to_output.at(axis_input).get_node_shared_ptr());
        auto indices_constant = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
            pattern_to_output.at(indices_input).get_node_shared_ptr());

        if (!gather || !axis_constant || !indices_constant) {
            return false;
        }

        auto indices = indices_constant->cast_vector<int64_t>();
        if (indices.size() != 1 || indices[0] >= 0) {
            return false;
        }

        auto axis = axis_constant->cast_vector<int64_t>();
        if (axis.size() != 1) {
            return false;
        }

        auto axis_value = axis[0];

        // normalize `axis` value if it is negative
        if (axis_value < 0) {
            axis_value = axis_value + data.get_partial_shape().rank().get_length();
        }

        if (data.get_partial_shape().rank().get_length() < axis_value) {
            return false;
        }

        // check `axis` dimension of data tensor is static
        if (!data.get_partial_shape()[axis_value].is_static()) {
            return false;
        }

        auto input_type = indices_constant->get_element_type();
        auto shape_of = std::make_shared<ngraph::opset7::ShapeOf>(data, input_type);
        auto input_gather = std::make_shared<ngraph::opset7::Gather>(
            shape_of,
            ngraph::opset7::Constant::create(input_type, Shape{}, {axis_value}),
            ngraph::opset7::Constant::create(input_type, Shape{}, {0}));

        std::shared_ptr<Node> add = std::make_shared<ngraph::opset7::Add>(input_gather, indices_constant);
        if (auto folded_const = ngraph::get_constant_from_source(add))
            add = folded_const;
        gather->input(1).replace_source_output(add);

        ngraph::copy_runtime_info(gather, {shape_of, input_gather, add});
        MATCHER_SCOPE_ENABLE(GatherNegativeConstIndicesNormalize);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather_node, matcher_name);
    register_matcher(m, callback);
}
