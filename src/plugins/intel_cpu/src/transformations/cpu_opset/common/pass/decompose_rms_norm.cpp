// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decompose_rms_norm.hpp"
#include "itt.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/core/rt_info.hpp"
#include "ov_ops/rms.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

DecomposeRMSNorm::DecomposeRMSNorm() {
    MATCHER_SCOPE(DecomposeRMSNorm);
    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::internal::RMS>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node = std::dynamic_pointer_cast<ov::op::internal::RMS>(
            pattern_to_output.at(pattern_node).get_node_shared_ptr());

        if (node == nullptr || transformation_callback(node)) {
            return false;
        }
        auto data = node->get_input_node_shared_ptr(0);
        auto data_precision = node->get_input_element_type(0);
        auto scale = node->get_input_node_shared_ptr(1);

        auto power_const = ov::opset10::Constant::create(data_precision, {}, std::vector<float>{2.f});
        auto power = std::make_shared<ov::opset10::Power>(data, power_const);
        auto mean_axes = ov::opset10::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::opset10::ReduceMean>(power, mean_axes, true);
        auto eps = ov::opset10::Constant::create(data_precision, {}, {node->get_epsilon()});
        auto add_eps = std::make_shared<ov::opset10::Add>(mean, eps);
        auto sqrt = std::make_shared<ov::opset10::Sqrt>(add_eps);
        auto div_const = ov::opset10::Constant::create(data_precision, {}, {-1});
        auto div = std::make_shared<ov::opset10::Power>(sqrt, div_const);
        auto mul1 = std::make_shared<ov::opset10::Multiply>(data, div);
        auto mul2 = std::make_shared<ov::opset10::Multiply>(scale, mul1);

        ov::replace_node(node, mul2);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, matcher_name);
    register_matcher(m, callback);
}

}   // namespace intel_cpu
}   // namespace ov
