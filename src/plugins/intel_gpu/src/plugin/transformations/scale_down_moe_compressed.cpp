// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scale_down_moe_compressed.hpp"

#include "intel_gpu/op/moe_compressed.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

ScaleDownMOECompressed::ScaleDownMOECompressed(float scale_factor, ov::element::Type scaled_prec) {
    auto moe_m = ov::pass::pattern::wrap_type<ov::intel_gpu::op::MOECompressed>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto moe = ov::as_type_ptr<ov::intel_gpu::op::MOECompressed>(m.get_match_root());
        if (!moe)
            return false;

        if (transformation_callback(moe))
            return false;

        const auto& config = moe->get_config();
        // Only GEMM2_BIAS_SWIGLU_CLAMP has the bias inputs that require scaling.
        if (config.expert_type != ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP)
            return false;

        // Idempotence: skip ops already scaled.
        if (moe->get_scale_factor() > 0.0f)
            return false;

        auto insert_scale_down = [&](size_t input_idx) {
            const std::vector<float> scale_down_value = {1.f / scale_factor};
            const auto src = moe->input(input_idx).get_source_output();

            auto scale_const = std::make_shared<ov::op::v0::Constant>(src.get_element_type(),
                                                                      ov::Shape(),
                                                                      scale_down_value);
            auto scale_down = std::make_shared<ov::op::v1::Multiply>(src, scale_const);
            scale_down->set_friendly_name(moe->get_friendly_name() + "_scale_down_in" +
                                          std::to_string(input_idx));
            ov::copy_runtime_info(moe, scale_down);

            ov::Output<ov::Node> new_src = scale_down->output(0);
            if (new_src.get_element_type() != scaled_prec) {
                auto convert_prec = std::make_shared<ov::op::v0::Convert>(new_src, scaled_prec);
                ov::copy_runtime_info(moe, convert_prec);
                new_src = convert_prec->output(0);
            }
            moe->input(input_idx).replace_source_output(new_src);
        };

        // Input layout for MOECompressed / GEMM2_BIAS_SWIGLU_CLAMP (see moe.cpp CreateMOECompressedOp):
        //   0: hidden_states
        //   1: routing_weights
        //   2: topk_idx
        //   3: w_up
        //   4: scale_up
        //   [5: zp_up, if has_zp]
        //   <bias_up>      -- index 5 (no zp) or 6 (has_zp)
        //   <w_down>       -- bias_up + 1
        //   <scale_down>   -- bias_up + 2
        //   [zp_down,      -- bias_up + 3 if has_zp]
        //   <bias_down>    -- last input
        const size_t bias_up_idx = config.has_zp ? 6u : 5u;
        const size_t bias_down_idx = moe->get_input_size() - 1u;

        // Scale down hidden_states (activation input) and both biases.
        insert_scale_down(0);
        insert_scale_down(bias_up_idx);
        insert_scale_down(bias_down_idx);

        const std::vector<float> scale_up_value = {scale_factor};
        std::set<ov::Input<ov::Node>> target_inputs = moe->get_output_target_inputs(0);
        auto scale_up = register_new_node<ov::op::v1::Multiply>(
            moe->output(0),
            std::make_shared<ov::op::v0::Constant>(moe->output(0).get_element_type(),
                                           ov::Shape(),
                                           scale_up_value));
        scale_up->set_friendly_name(moe->get_friendly_name() + "_scale_up");
        ov::copy_runtime_info(moe, scale_up);
        for (auto& in : target_inputs) {
            in.replace_source_output(scale_up);
        }

        moe->set_scale_factor(scale_factor);
        moe->validate_and_infer_types();

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_m, "ScaleDownMOECompressed");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
