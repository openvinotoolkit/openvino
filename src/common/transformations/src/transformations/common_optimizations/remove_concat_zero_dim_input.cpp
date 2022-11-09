// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_concat_zero_dim_input.hpp"

#include <algorithm>
#include <memory>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::RemoveConcatZeroDimInput::RemoveConcatZeroDimInput() {
    MATCHER_SCOPE(RemoveConcatZeroDimInput);
    auto concat_pattern = pattern::wrap_type<opset8::Concat>();
    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto concat = m.get_match_root();
        const auto& rt_info = concat->get_rt_info();
        if (rt_info.count(DisableRemoveConcatZeroDimInput::get_type_info_static())) {
            return false;
        }

        auto concat_inputs = concat->input_values();
        concat_inputs.erase(
            std::remove_if(
                concat_inputs.begin(),
                concat_inputs.end(),
                [](const Output<Node>& input) {
                    const auto& in_shape = input.get_partial_shape();
                    if (in_shape.rank().is_static()) {
                        return std::any_of(std::begin(in_shape), std::end(in_shape), [](const ov::Dimension& dim) {
                            if (dim.is_static() && dim.get_length() == 0) {
                                return true;
                            }
                            return false;
                        });
                    }
                    return false;
                }),
            concat_inputs.end());

        bool inputs_removed = concat->get_input_size() > concat_inputs.size();
        if (inputs_removed) {
            concat->set_arguments(concat_inputs);
        }
        return inputs_removed;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_pattern, matcher_name);
    this->register_matcher(m, callback);
}

void ov::pass::disable_remove_concat_zerodim_input(const std::shared_ptr<Node>& node) {
    node->get_rt_info().emplace(DisableRemoveConcatZeroDimInput::get_type_info_static(),
                                DisableRemoveConcatZeroDimInput{});
}

void ov::pass::enable_remove_concat_zerodim_input(const std::shared_ptr<Node>& node) {
    node->get_rt_info().erase(DisableRemoveConcatZeroDimInput::get_type_info_static());
}

bool ov::pass::remove_concat_zerodim_input_is_disabled(const std::shared_ptr<Node>& node) {
    return node->get_rt_info().count(DisableRemoveConcatZeroDimInput::get_type_info_static());
}
