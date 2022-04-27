// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swap_convert_transpose.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::SwapConvertTranspose, "SwapConvertTranspose", 0);

ov::intel_cpu::SwapConvertTranspose::SwapConvertTranspose() {
    MATCHER_SCOPE(SwapConvertTranspose);
    ngraph::element::TypeVector param_precisions{ ngraph::element::i8, ngraph::element::u8 };
    auto input_m = ngraph::pattern::wrap_type<ngraph::op::v0::Parameter>(ngraph::pattern::type_matches_any(param_precisions));
    auto convert_m = ngraph::pattern::wrap_type<ngraph::op::v0::Convert>({input_m}, ngraph::pattern::type_matches(ngraph::element::f32));
    auto transpose_m = ngraph::pattern::wrap_type<ngraph::op::v1::Transpose>({convert_m, ngraph::pattern::any_input()});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        // Swap
        // Input -> [i8/u8] -> Convert -> [f32] -> Transpose -> [f32]
        // to
        // Input -> [i8/u8] -> Transpose -> [i8/u8] -> Convert -> [f32]
        const auto& pattern_map = m.get_pattern_value_map();
        auto convert = pattern_map.at(convert_m).get_node_shared_ptr();
        auto transpose = pattern_map.at(transpose_m).get_node_shared_ptr();

        ngraph::OutputVector transposeInputs = transpose->input_values();
        transposeInputs[0] = convert->input_value(0);
        auto newTranspose = transpose->clone_with_new_inputs(transposeInputs);
        ngraph::copy_runtime_info(transpose, newTranspose);
        newTranspose->set_friendly_name(transpose->get_friendly_name());

        ngraph::OutputVector convertInputs = convert->input_values();
        convertInputs[0] = newTranspose;
        auto newConvert = convert->clone_with_new_inputs(convertInputs);
        ngraph::replace_node(transpose, newConvert);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_m, matcher_name);
    this->register_matcher(m, callback);
}
