// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_concat_zero_dim_input.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::RemoveConcatZeroDimInput, "RemoveConcatZeroDimInput", 0);

ngraph::pass::RemoveConcatZeroDimInput::RemoveConcatZeroDimInput() {
    MATCHER_SCOPE(RemoveConcatZeroDimInput);
    auto concat_pattern = pattern::wrap_type<opset8::Concat>();
    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto concat = std::dynamic_pointer_cast<opset8::Concat>(m.get_match_root());
        OutputVector correct_inputs;
        bool replacement_expected = false;
        for (const auto& input : concat->input_values()) {
            bool current_in_correct = true;
            const auto& in_shape = input.get_partial_shape();
            if (in_shape.rank().is_static()) {
                for (const auto& dim : in_shape) {
                    if (dim.is_static() && dim.get_length() == 0) {
                        replacement_expected = true;
                        current_in_correct = false;
                    }
                }
            }
            if (current_in_correct)
                correct_inputs.push_back(input);
        }
        if (!replacement_expected)
            return false;

        auto new_concat = std::make_shared<opset8::Concat>(correct_inputs, concat->get_axis());
        new_concat->set_friendly_name(concat->get_friendly_name());
        ngraph::copy_runtime_info(concat, new_concat);
        ngraph::replace_node(concat, new_concat);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_pattern, matcher_name);
    this->register_matcher(m, callback);
}
