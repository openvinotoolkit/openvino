// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swap_convert_transpose.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::intel_cpu::SwapConvertTranspose::SwapConvertTranspose() {
    MATCHER_SCOPE(SwapConvertTranspose);
    ov::element::TypeVector param_precisions{ov::element::i8, ov::element::u8};
    auto input_m =
        ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(ov::pass::pattern::type_matches_any(param_precisions));
    auto convert_m =
        ov::pass::pattern::wrap_type<ov::op::v0::Convert>({input_m}, ov::pass::pattern::type_matches(ov::element::f32));
    auto transpose_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({convert_m, ov::pass::pattern::any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        // Swap
        // Input -> [i8/u8] -> Convert -> [f32] -> Transpose -> [f32]
        // to
        // Input -> [i8/u8] -> Transpose -> [i8/u8] -> Convert -> [f32]
        const auto& pattern_map = m.get_pattern_value_map();
        auto convert = pattern_map.at(convert_m).get_node_shared_ptr();
        auto transpose = pattern_map.at(transpose_m).get_node_shared_ptr();

        if (convert->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        ov::OutputVector transposeInputs = transpose->input_values();
        transposeInputs[0] = convert->input_value(0);
        auto newTranspose = transpose->clone_with_new_inputs(transposeInputs);
        newTranspose->set_friendly_name(transpose->get_friendly_name() + "_original");

        ov::OutputVector convertInputs = convert->input_values();
        convertInputs[0] = newTranspose;
        auto newConvert = convert->clone_with_new_inputs(convertInputs);
        ov::replace_node(transpose, newConvert);
        newConvert->set_friendly_name(transpose->get_friendly_name());

        ov::copy_runtime_info(transpose, {newTranspose, newConvert});
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_m, matcher_name);
    this->register_matcher(m, callback);
}
