// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/isolate_precision_sensitive_outputs.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::frontend::gguf::pass {

IsolatePrecisionSensitiveOutputs::IsolatePrecisionSensitiveOutputs() {
    MATCHER_SCOPE(IsolatePrecisionSensitiveOutputs);
    const auto protected_fp32 = ov::pass::pattern::any_input([](const ov::Output<ov::Node>& value) {
        return value.get_element_type() == ov::element::f32 &&
               ov::is_conversion_disabled(value.get_node_shared_ptr(), ov::element::f16);
    });
    const auto split = ov::pass::pattern::wrap_type<ov::op::v1::Split>(
        {protected_fp32, ov::pass::pattern::any_input()},
        [](const ov::Output<ov::Node>& value) {
            return !ov::is_conversion_disabled(value.get_node_shared_ptr(), ov::element::f16);
        });
    const ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& matcher) {
        const auto node = matcher.get_match_root();
        if (transformation_callback(node))
            return false;
        const auto input = matcher.get_pattern_value_map().at(protected_fp32);
        const auto boundary = std::make_shared<ov::op::v0::Convert>(input, input.get_element_type());
        ov::copy_runtime_info(node, boundary);
        node->input(0).replace_source_output(boundary);
        return true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(split, matcher_name), callback);
}

}  // namespace ov::frontend::gguf::pass
