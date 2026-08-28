// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_compression.hpp"

#include <memory>
#include <vector>

#include "ov_ops/rms.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

DisableFP16CompForGemma3RMSPattern::DisableFP16CompForGemma3RMSPattern() {
    using namespace ov::pass::pattern;

    auto const_or_convert = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{
        wrap_type<ov::op::v0::Constant>(),
        wrap_type<ov::op::v0::Convert>({wrap_type<ov::op::v0::Constant>()})
    });

    auto add_m = wrap_type<ov::op::v1::Add>({any_input(), any_input()}, type_matches(element::f32));
    auto rms_post_m = wrap_type<ov::op::internal::RMS>({any_input(), const_or_convert}, type_matches(element::f32));
    auto add_1_m = wrap_type<ov::op::v1::Add>({add_m, rms_post_m}, type_matches(element::f32));
    auto rms_m = wrap_type<ov::op::internal::RMS>({add_1_m, const_or_convert}, type_matches(element::f32));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto rms = ov::as_type_ptr<ov::op::internal::RMS>(pattern_map.at(rms_m).get_node_shared_ptr());
        if (!rms || transformation_callback(rms))
            return false;

        auto rms_post = pattern_map.at(rms_post_m).get_node_shared_ptr();
        if (rms_post)
            ov::disable_conversion(rms_post, element::f16);

        ov::disable_conversion(rms, element::f16);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rms_m, "DisableFP16CompForGemma3RMSPattern");
    register_matcher(m, callback);
}

DisableFP16CompForGatedResidualPattern::DisableFP16CompForGatedResidualPattern() {
    using namespace ov::pass::pattern;

    auto mvn_m = wrap_type<ov::op::v6::MVN>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto mvn = ov::as_type_ptr<ov::op::v6::MVN>(m.get_match_root());
        if (!mvn || mvn->get_output_element_type(0) != element::f32 || transformation_callback(mvn))
            return false;

        auto residual_add = ov::as_type_ptr<ov::op::v1::Add>(mvn->get_input_node_shared_ptr(0));
        if (!residual_add)
            return false;

        bool multiply_found = false;
        for (const auto& input : residual_add->input_values()) {
            const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(input.get_node_shared_ptr());
            if (!multiply)
                continue;

            multiply_found = true;
            for (const auto& multiply_input : multiply->input_values()) {
                const auto producer = multiply_input.get_node_shared_ptr();
                ov::disable_conversion(producer, element::f16);

                const auto linear_add = ov::as_type_ptr<ov::op::v1::Add>(producer);
                if (!linear_add)
                    continue;

                for (const auto& linear_input : linear_add->input_values()) {
                    const auto linear_producer = linear_input.get_node_shared_ptr();
                    if (ov::is_type<ov::op::v0::MatMul>(linear_producer))
                        ov::disable_conversion(linear_producer, element::f16);
                }
            }

            ov::disable_conversion(multiply, element::f16);
        }
        if (!multiply_found)
            return false;

        ov::disable_conversion(residual_add, element::f16);
        ov::disable_conversion(mvn, element::f16);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mvn_m, "DisableFP16CompForGatedResidualPattern");
    register_matcher(m, callback);
}

DisableFP16CompForDirectMultiplySinCos::DisableFP16CompForDirectMultiplySinCos() {
    using namespace ov::pass::pattern;

    auto multiply = wrap_type<ov::op::v1::Multiply>({any_input(), any_input()}, type_matches(element::f32));
    auto sin = wrap_type<ov::op::v0::Sin>({multiply}, type_matches(element::f32));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto multiply_node = pattern_map.at(multiply).get_node_shared_ptr();
        const auto sin_node = pattern_map.at(sin).get_node_shared_ptr();
        if (transformation_callback(sin_node))
            return false;

        std::vector<std::shared_ptr<ov::op::v0::Cos>> cos_nodes;
        for (const auto& user : multiply_node->get_users()) {
            if (const auto cos_node = ov::as_type_ptr<ov::op::v0::Cos>(user))
                cos_nodes.push_back(cos_node);
        }
        if (cos_nodes.empty())
            return false;

        for (const auto& input : multiply_node->input_values())
            ov::disable_conversion(input.get_node_shared_ptr(), element::f16);
        ov::disable_conversion(multiply_node, element::f16);
        ov::disable_conversion(sin_node, element::f16);
        for (const auto& cos_node : cos_nodes)
            ov::disable_conversion(cos_node, element::f16);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin, "DisableFP16CompForDirectMultiplySinCos");
    register_matcher(m, callback);
}

DisableFP16ComForGPTOSSROPEPattern::DisableFP16ComForGPTOSSROPEPattern() {
    using namespace ov::pass::pattern;

    // For the GPT-OSS pattern.
    auto freq_const = wrap_type<ov::op::v0::Constant>();
    auto broadcast_freq = wrap_type<ov::op::v3::Broadcast>({freq_const, any_input()});

    // Position IDs.
    auto unsqueeze_pos_id = wrap_type<ov::op::v0::Unsqueeze>({any_input(), any_input()});
    auto convert_pos_id_to_f16 = wrap_type<ov::op::v0::Convert>({unsqueeze_pos_id});
    auto matmul_freq_pos_id = wrap_type<ov::op::v0::MatMul>({broadcast_freq, convert_pos_id_to_f16});
    auto transpose = wrap_type<ov::op::v1::Transpose>({matmul_freq_pos_id, any_input()});
    auto sin = wrap_type<ov::op::v0::Sin>({transpose});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sin_node = ov::as_type_ptr<ov::op::v0::Sin>(pattern_map.at(sin).get_node_shared_ptr());
        if (!sin_node || transformation_callback(sin_node))
            return false;
        auto freq_const_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(freq_const).get_node_shared_ptr());
        ov::disable_conversion(freq_const_node, element::f16);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin, "DisableFP16ComForGPTOSSROPEPattern");
    register_matcher(m, callback);
}

// See the class documentation for the F0 oscillator pattern and motivation.
DisableFP16CompCumSumSinGen::DisableFP16CompCumSumSinGen() {
    using namespace ov::pass::pattern;

    auto cumsum_m = wrap_type<ov::op::v0::CumSum>({any_input(), any_input()});
    auto mul1_m = wrap_type<ov::op::v1::Multiply>({cumsum_m, any_input()});
    auto transpose2_m = wrap_type<ov::op::v1::Transpose>({mul1_m, any_input()});
    auto mul2_m = wrap_type<ov::op::v1::Multiply>({transpose2_m, any_input()});
    auto interpolate_m =
        wrap_type<ov::op::v4::Interpolate, ov::op::v11::Interpolate>({mul2_m, any_input(), any_input()});
    auto transpose3_m = wrap_type<ov::op::v1::Transpose>({interpolate_m, any_input()});
    auto sin_m = wrap_type<ov::op::v0::Sin>({transpose3_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sin_node = pattern_map.at(sin_m).get_node_shared_ptr();
        if (transformation_callback(sin_node))
            return false;

        auto cumsum_node = pattern_map.at(cumsum_m).get_node_shared_ptr();

        // Also tag the producer feeding CumSum's first input.
        auto cumsum_input = cumsum_node->input_value(0).get_node_shared_ptr();
        if (cumsum_input)
            ov::disable_conversion(cumsum_input, element::f16);

        for (const auto& key : {cumsum_m, mul1_m, transpose2_m, mul2_m, interpolate_m, transpose3_m, sin_m})
            ov::disable_conversion(pattern_map.at(key).get_node_shared_ptr(), element::f16);
        return true;
    };

    auto m = std::make_shared<Matcher>(sin_m, "DisableFP16CompCumSumSinGen");
    register_matcher(m, callback);
}

DisableFP16ComSinGenPatternForHiFiGAN::DisableFP16ComSinGenPatternForHiFiGAN() {
    using namespace ov::pass::pattern;
    using ov::op::v0::Sin;
    using ov::op::v1::Multiply;
    using ov::op::v1::Transpose;

    // SineGen of HiFiGAN
    // (https://github.com/FunAudioLLM/CosyVoice/blob/1dcc59676fe3fa863f983ab7820e481560c73be7/cosyvoice/hifigan/generator.py#L157-L189)
    // can produce inf in fp16 because of large input value multiplication
    // (for example, hop_length=480 makes Multiply x480). Keep the path from
    // Multiply x480 to Sin in FP32 to avoid the overflow.
    auto multiply = wrap_type<Multiply>();
    // This pass is called after ConvertToInterpolateV4 passes. Keep all
    // currently supported variants here for compatibility with existing IRs.
    auto interpolate_v0 = wrap_type<ov::op::v0::Interpolate>({multiply, any_input()});
    auto interpolate_v4 = wrap_type<ov::op::v4::Interpolate>({multiply, any_input(), any_input()});
    auto interpolate_v4_with_axes = wrap_type<ov::op::v4::Interpolate>({multiply, any_input(), any_input(), any_input()});
    auto interpolate_v11 = wrap_type<ov::op::v11::Interpolate>({multiply, any_input()});
    auto interpolate_v11_with_axes = wrap_type<ov::op::v11::Interpolate>({multiply, any_input(), any_input()});
    auto interpolate = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{
        interpolate_v0,
        interpolate_v4,
        interpolate_v4_with_axes,
        interpolate_v11,
        interpolate_v11_with_axes
    });
    auto transpose = wrap_type<Transpose>({interpolate, any_input()});
    auto sin = wrap_type<Sin>({transpose});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto sin_node = pattern_map.at(sin).get_node_shared_ptr();
        auto transpose_node = pattern_map.at(transpose).get_node_shared_ptr();
        auto interpolate_node = pattern_map.at(interpolate).get_node_shared_ptr();
        auto multiply_node = pattern_map.at(multiply).get_node_shared_ptr();

        if (transformation_callback(sin_node))
            return false;

        for (const auto& node : {multiply_node, interpolate_node, transpose_node, sin_node})
            ov::disable_conversion(node, element::f16);
        return true;
    };

    auto m = std::make_shared<Matcher>(sin, "DisableFP16ComSinGenPatternForHiFiGAN");
    register_matcher(m, callback);
}

bool DisableFP16Compression::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager(get_pass_config(), "DisableFP16Compression");
    manager.set_per_pass_validation(false);

    manager.register_pass<DisableFP16CompForGemma3RMSPattern>();
    manager.register_pass<DisableFP16CompForGatedResidualPattern>();
    manager.register_pass<DisableFP16ComForGPTOSSROPEPattern>();
    manager.register_pass<DisableFP16CompForDirectMultiplySinCos>();
    manager.register_pass<DisableFP16CompCumSumSinGen>();

    // HiFiGAN matches a strict suffix of the CumSumSinGen chain. Skip it when
    // the same Sin was already marked by the more specific matcher above.
    get_pass_config()->set_callback<DisableFP16ComSinGenPatternForHiFiGAN>(
        [](const std::shared_ptr<const ov::Node>& node) {
            return ov::is_conversion_disabled(node, ov::element::f16);
        });
    manager.register_pass<DisableFP16ComSinGenPatternForHiFiGAN>();

    return manager.run_passes(model);
}

}  // namespace ov::intel_gpu
