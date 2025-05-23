// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_dequantization_fp16.hpp"

#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

ConvertDequantizationFP16::ConvertDequantizationFP16(const element::TypeVector& precisions) {
    add_matcher<ConvertDequantizationFP16Matcher>(precisions);
}

ConvertDequantizationFP16Matcher::ConvertDequantizationFP16Matcher(const element::TypeVector& precisions) {
    using namespace ov::op;
    using namespace ov::pass::pattern;

    auto input_pattern = any_input(type_matches_any(precisions));
    auto convert_pattern = wrap_type<v0::Convert>({input_pattern}, consumers_count(1));

    // zero points:
    auto zp_pattern = any_input();
    auto zp_convert_pattern = optional<v0::Convert>(zp_pattern);
    auto zp_reshape_pattern = optional<v1::Reshape, v0::Unsqueeze>({zp_convert_pattern, any_input()});
    auto subtract_pattern = optional<v1::Subtract>({convert_pattern, zp_reshape_pattern});

    // scale:
    auto scale_pattern = any_input();
    auto scale_convert_pattern = optional<v0::Convert>(scale_pattern);
    auto scale_reshape_pattern = optional<v1::Reshape, v0::Unsqueeze>({scale_convert_pattern, any_input()});
    auto multiply_pattern = wrap_type<v1::Multiply>({subtract_pattern, scale_reshape_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto multiply = m.get_match_root();
        
        auto multiply_users = multiply->get_users();
        for (const auto& user : multiply_users) {
            for (size_t idx = 0; idx < user->inputs().size(); ++idx) {
                if (user->get_input_node_shared_ptr(idx) == multiply) {
                    auto new_convert = std::make_shared<v0::Convert>(multiply, multiply->get_output_element_type(0));
                    ov::mark_as_precision_sensitive(new_convert->input(0));
                    user->input(idx).replace_source_output(new_convert->output(0));
                }
            }
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(multiply_pattern, "ConvertDequantizationFP16Matcher");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
