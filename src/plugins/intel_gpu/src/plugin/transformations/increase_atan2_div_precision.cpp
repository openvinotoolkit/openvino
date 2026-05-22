// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_atan2_div_precision.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/select.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "utils.hpp"

namespace ov::intel_gpu {

IncreaseAtan2DivPrecision::IncreaseAtan2DivPrecision() {
    using namespace ov::pass::pattern;

    // After common ConvertDivide pass, Divide(a, b) is rewritten to
    // Multiply(a, Power(b, -1)). Match the post-ConvertDivide form.
    auto mul_m = wrap_type<ov::op::v1::Multiply>({any_input(), any_input()});
    auto atan_m = wrap_type<ov::op::v0::Atan>({mul_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto mul_node = pattern_map.at(mul_m).get_node_shared_ptr();
        auto atan_node = pattern_map.at(atan_m).get_node_shared_ptr();

        if (transformation_callback(mul_node)) {
            return false;
        }

        // Locate the Power(b, Constant) input of the Multiply.
        std::shared_ptr<ov::op::v1::Power> power_node;
        for (size_t i = 0; i < mul_node->get_input_size(); ++i) {
            auto in = mul_node->get_input_node_shared_ptr(i);
            auto p = ov::as_type_ptr<ov::op::v1::Power>(in);
            if (!p)
                continue;
            if (!ov::as_type_ptr<ov::op::v0::Constant>(p->get_input_node_shared_ptr(1)))
                continue;
            power_node = p;
            break;
        }
        if (!power_node)
            return false;

        const auto desired_et = ov::element::f32;
        const auto original_et = mul_node->get_output_element_type(0);
        if (original_et == desired_et)
            return false;
        if (!original_et.is_real())
            return false;

        // Capture the original divisor (b in Power(b, -1)) before promotion so
        // the imag==0 guard below compares against the source values.
        const auto imag_src = power_node->input_value(0);

        size_t input_idx = 0;
        bool changed = insert_converts_before_if_needed(power_node, desired_et, input_idx);
        changed = insert_converts_before_if_needed(mul_node, desired_et, input_idx) || changed;
        if (!changed)
            return false;

        // Guard against imag == 0: Power(0, -1) = +Inf, and real * Inf = NaN
        // when real is also 0. atan2(0, 0) is defined as 0 in PyTorch, so
        // replace the Multiply output with 0 wherever imag is exactly zero.
        power_node->revalidate_and_infer_types();
        mul_node->revalidate_and_infer_types();
        const auto mul_out_et = mul_node->get_output_element_type(0);
        auto zero_cmp = std::make_shared<ov::op::v0::Constant>(imag_src.get_element_type(), ov::Shape{}, std::vector<float>{0.0f});
        auto is_zero = std::make_shared<ov::op::v1::Equal>(imag_src, zero_cmp);
        auto zero_repl = std::make_shared<ov::op::v0::Constant>(mul_out_et, ov::Shape{}, std::vector<float>{0.0f});
        auto guarded = std::make_shared<ov::op::v1::Select>(is_zero, zero_repl, mul_node->output(0));
        guarded->set_friendly_name(mul_node->get_friendly_name() + "_imag_zero_guard");
        ov::copy_runtime_info(mul_node, guarded);
        for (auto& target_input : mul_node->output(0).get_target_inputs()) {
            if (target_input.get_node() == guarded.get())
                continue;
            target_input.replace_source_output(guarded->output(0));
        }

        size_t output_idx = 0;
        insert_converts_after_if_needed(atan_node, original_et, output_idx);
        return true;
    };

    auto m = std::make_shared<Matcher>(atan_m, "IncreaseAtan2DivPrecision");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
