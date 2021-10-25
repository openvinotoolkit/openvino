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
        auto concat_inputs = concat->input_values();
        bool pass_applied = false;
        concat_inputs.erase(std::remove_if(concat_inputs.begin(), concat_inputs.end(),
            [&pass_applied](const Output<Node>& input){
            const auto& in_shape = input.get_partial_shape();
                if (in_shape.rank().is_static()) {
                    for (const auto& dim : in_shape) {
                        if (dim.is_static() && dim.get_length() == 0) {
                            pass_applied = true;
                            return true;
                        }
                    }
                }
                return false;
            }), concat_inputs.end());
        if (pass_applied) {
            concat->set_arguments(concat_inputs);
        }
        return pass_applied;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_pattern, matcher_name);
    this->register_matcher(m, callback);
}
