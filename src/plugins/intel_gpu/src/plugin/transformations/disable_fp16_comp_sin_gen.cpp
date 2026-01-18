// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_sin_gen.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace ov::intel_gpu {
bool DisableFP16ComSinGenPatternForHiFiGAN::insert_converts_before_if_needed(const std::shared_ptr<ov::Node>& node,
                                                                const ov::element::Type desired_et, size_t& input_idx) {
    bool is_changed = false;
    for (const auto& input : node->inputs()) {
        const auto& incoming_output = input.get_source_output();
        const auto& incoming_node = incoming_output.get_node_shared_ptr();
        const auto input_et = incoming_output.get_element_type();

        if (input_et == desired_et)
            continue;

        auto in_convert = ov::as_type_ptr<ov::op::v0::Convert>(incoming_node);

        if (in_convert && in_convert->get_users().size() == 1 && input_et.bitwidth() <= desired_et.bitwidth()) {
            auto convert = std::make_shared<ov::op::v0::Convert>(incoming_node->input_value(0), desired_et);
            convert->set_friendly_name(in_convert->get_friendly_name() + "_increase_precision_" + std::to_string(input_idx));
            copy_runtime_info(incoming_node, convert);
            ov::replace_node(incoming_node, convert);
        } else {
            auto convert = std::make_shared<ov::op::v0::Convert>(incoming_output, desired_et);
            convert->set_friendly_name(incoming_node->get_friendly_name() + "_increase_precision_" + std::to_string(input_idx));
            copy_runtime_info(incoming_node, convert);
            input.replace_source_output(convert);
        }

        input_idx++;
        is_changed = true;
    }

    return is_changed;
}

void DisableFP16ComSinGenPatternForHiFiGAN::insert_converts_after_if_needed(const std::shared_ptr<ov::Node>& node,
                                                            const ov::element::Type original_et, size_t& output_idx) {
    for (const auto& output : node->outputs()) {
        for (const auto& out_inputs : output.get_target_inputs()) {
            auto out_node = out_inputs.get_node()->shared_from_this();

            auto convert = std::make_shared<ov::op::v0::Convert>(output, original_et);
            auto convert_name = out_node->get_friendly_name() + "_restore_precision_" + std::to_string(output_idx);
            convert->set_friendly_name(convert_name);
            copy_runtime_info(node, convert);
            out_inputs.replace_source_output(convert);
            output_idx++;
        }
    }
}

DisableFP16ComSinGenPatternForHiFiGAN::DisableFP16ComSinGenPatternForHiFiGAN() {
    using namespace ov::pass::pattern;

    // SineGen of HiFiGAN(https://github.com/FunAudioLLM/CosyVoice/blob/1dcc59676fe3fa863f983ab7820e481560c73be7/cosyvoice/hifigan/generator.py#L157-L189)
    // could make inf in fp16 because of large input value multiplication (e.g. hop_length=480 makes multiply x480)
    // So keep fp32 from Multiply x480 to Sin to avoid inf in fp16
    auto multiply = wrap_type<ov::op::v1::Multiply>();
    // This pass is called after ConvertToInterpolateV4 passes. So consider only v4 here.
    auto interpolate = wrap_type<ov::op::v4::Interpolate>({multiply, any_input(), any_input(), any_input()});
    auto transpose = wrap_type<ov::op::v1::Transpose>({interpolate, any_input()});
    auto sin = wrap_type<ov::op::v0::Sin>({transpose});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto sin_node = pattern_map.at(sin).get_node_shared_ptr();
        auto transpose_node = pattern_map.at(transpose).get_node_shared_ptr();
        auto interpolate_node = pattern_map.at(interpolate).get_node_shared_ptr();
        auto multiply_node = pattern_map.at(multiply).get_node_shared_ptr();

        if (transformation_callback(sin_node)) return false;

        const auto desired_et = ov::element::f32;
        const auto original_et = sin_node->get_output_element_type(0);

        if (original_et == desired_et) return false;

        size_t idx = 0;
        insert_converts_before_if_needed(multiply_node, desired_et, idx);
        ov::disable_fp16_compression(multiply_node);
        ov::disable_fp16_compression(interpolate_node);
        ov::disable_fp16_compression(transpose_node);
        ov::disable_fp16_compression(sin_node);
        insert_converts_after_if_needed(sin_node, original_et, idx);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin, "DisableFP16ComSinGenPatternForHiFiGAN");
    this->register_matcher(m, callback);
}
}  // namespace ov::intel_gpu
