// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/softmax_sr.hpp"

#include <memory>

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset4.hpp>

ngraph::pass::ReshapeSoftMaxReshape::ReshapeSoftMaxReshape() {
    auto reshape_label = ngraph::pattern::wrap_type<opset4::Reshape>();
    auto softmax_label = ngraph::pattern::wrap_type<opset4::Softmax>({reshape_label});
    auto reshape_back_const_label = ngraph::pattern::wrap_type<opset4::Constant>();
    auto reshape_back_label = ngraph::pattern::wrap_type<opset4::Reshape>({softmax_label, reshape_back_const_label});

    matcher_pass_callback callback = [=](pattern::Matcher &m) -> bool {
        const auto &pattern_to_output = m.get_pattern_value_map();

        const auto &const_reshape_pattern = std::dynamic_pointer_cast<opset4::Constant>(pattern_to_output.at(reshape_back_const_label).get_node_shared_ptr());
        if (!const_reshape_pattern)
            return false;

        const auto &reshape = pattern_to_output.at(reshape_label);
        const auto &reshape_back = pattern_to_output.at(reshape_back_label);

        const auto &pattern_input_shape = reshape.get_node_shared_ptr()->get_input_partial_shape(0);
        const auto &pattern_output_shape = PartialShape(const_reshape_pattern->cast_vector<int64_t>());
        if (pattern_input_shape.is_dynamic() || pattern_output_shape.is_dynamic() || !pattern_input_shape.compatible(pattern_output_shape))
            return false;

        const auto &shape_of = std::make_shared<opset4::ShapeOf>(reshape.get_node_shared_ptr()->get_input_source_output(0));
        copy_runtime_info(const_reshape_pattern, shape_of);
        shape_of->set_friendly_name(const_reshape_pattern->get_friendly_name());
        reshape_back.get_node_shared_ptr()->input_value(1).replace(shape_of->output(0));
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_back_label, "ReshapeSoftMaxReshape");
    register_matcher(m, callback);
}