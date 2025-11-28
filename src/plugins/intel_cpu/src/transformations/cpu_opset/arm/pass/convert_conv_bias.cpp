// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "convert_conv_bias.hpp"

#include <iostream>
#include <memory>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/op/util/op_types.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

ov::intel_cpu::ConvertConvolutionBias::ConvertConvolutionBias() {
    std::cout << "[ConvertConvolutionBias] Registering transformation" << std::endl;

    // Build pattern: Convolution -> Multiply -> Add with Constant
    auto conv_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>(
        {ov::pass::pattern::any_input(), ov::pass::pattern::any_input()});
    
    auto multiply_m = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>(
        {conv_m, ov::pass::pattern::any_input()});
    
    auto bias_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    
    auto add_m = ov::pass::pattern::wrap_type<ov::op::v1::Add>(
        {multiply_m, bias_const_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(m.get_match_root());
        if (!mul) {
            return false;
        }

        // Get input and weights element types
        const auto& pattern_map = m.get_pattern_value_map();
        auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(pattern_map.at(conv_m).get_node_shared_ptr());
        const auto input_et = conv->get_input_element_type(0);
        const auto weights_et = conv->get_input_element_type(1);
        
        std::cout << "[ConvertConvolutionBias] Input type: " << input_et 
                  << ", Weights type: " << weights_et << std::endl;
        
        // Check if pattern matches one of the supported cases:
        // 1. u8 source, u8 or i8 weights
        // 2. i8 source, i8 weights
        bool is_acl_supported_case = (input_et == ov::element::u8 &&
                                      ov::intel_cpu::any_of(weights_et, ov::element::u8, ov::element::i8)) ||
                                     (input_et == ov::element::i8 && weights_et == ov::element::i8);
        
        if (!is_acl_supported_case) {
            std::cout << "[ConvertConvolutionBias] NOT APPLIED: Unsupported element type combination" << std::endl;
            return false;
        }

        //std::cout << mul->get_output_target_inputs(0);
        for (const auto& child : mul->get_output_target_inputs(0)) {
            if (auto add = ov::as_type<ov::op::v1::Add>(child.get_node())) {
                if (ov::op::util::is_on_path<ov::op::v0::Constant>(add->input_value(1))) {
                    std::cout << "[ConvertConvolutionBias] Found Add with Constant input" << std::endl;
                    if (add->input_value(1).get_node()->get_output_element_type(0) == ov::element::i32) {
                        // constant is already i32 - no transformation needed
                        break;
                    }
                    std::cout << "[ConvertConvolutionBias] Converting bias Constant to i32" << std::endl;
                    auto bias_const = ov::as_type_ptr<ov::op::v0::Constant>(add->input_value(1).get_node_shared_ptr());
                    auto convert_to_i32 = std::make_shared<ov::op::v0::Convert>(bias_const, ov::element::i32);
                    add->input(1).replace_source_output(convert_to_i32->output(0));
                    ov::copy_runtime_info(bias_const, convert_to_i32);
                    std::cout << "[ConvertConvolutionBias] APPLIED: Inserted Convert(i32) before Add" << std::endl;

                    return true;
                }
            }
        }
        
        return false;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(/*add_m*/multiply_m, "ConvertConvolutionBias");
    register_matcher(matcher, callback);
}
