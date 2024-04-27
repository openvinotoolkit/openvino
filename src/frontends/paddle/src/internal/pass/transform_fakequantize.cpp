// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/pass/transform_fakequantize.hpp"

#include "default_opset.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::frontend::paddle::op::default_opset;
using namespace ov;
using namespace ov::pass;
using namespace ov::frontend::paddle::op;

/*
                                  zero_point
                                      /
                        input    convert     scale
                           \        /         /
                            subtract  Multiply
   quantize_linear   ==>>        \      /
                                  Divide
                                     \
                                      Round
                                         \
                                          Clamp
                                             \
  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _   => FakeQuantize
                                               \           zero_point
                                                \              /
                                                Convert   Convert   scale
   dequantize_linear  ==>>                          \      /        /
                                                   Subtract    Multiply
                                                       \        /
                                                        Multiply
*/
ov::frontend::paddle::pass::TransformFakeQuantize::TransformFakeQuantize() {
    const auto input_label = pattern::any_input();
    const auto q_zp_label = pattern::any_input();
    // quantize phase
    const auto q_zp_cvt_label = pattern::wrap_type<Convert>({q_zp_label});
    const auto q_sub_label = pattern::wrap_type<Subtract>({input_label, q_zp_cvt_label});
    const auto q_real_scale_label = pattern::wrap_type<Multiply>();
    const auto div_label = pattern::wrap_type<Divide>({q_sub_label, q_real_scale_label});
    const auto round_label = pattern::wrap_type<Round>({div_label});
    const auto q_clamp_label = pattern::wrap_type<Clamp>({round_label});
    // dequantize phase
    const auto dq_cvt_label = pattern::wrap_type<Convert>({q_clamp_label});
    const auto dq_zp_label = pattern::any_input();
    const auto dq_zp_cvt_label = pattern::wrap_type<Convert>({dq_zp_label});
    const auto dq_sub_label = pattern::wrap_type<Subtract>({dq_cvt_label, dq_zp_cvt_label});
    const auto dq_real_scale_label = pattern::wrap_type<Multiply>();
    const auto output_label = pattern::wrap_type<Multiply>({dq_sub_label, dq_real_scale_label});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        // get the input
        const auto& sub_node = opsMap.at(q_sub_label).get_node_shared_ptr();
        if (!sub_node->get_input_node_shared_ptr(0)) {
            return false;
        }
        const auto& input_item = sub_node->get_input_source_output(0);

        // prepare for replace
        const auto& output_node = opsMap.at(output_label).get_node_shared_ptr();

        // check round mode
        // Fallback to the PDPD FE if the round_mode is HALF_AWAY_FROM_ZERO.
        const auto& round_node_cast = std::dynamic_pointer_cast<Round>(opsMap.at(round_label).get_node_shared_ptr());
        if (!round_node_cast || round_node_cast->get_mode() != Round::RoundMode::HALF_TO_EVEN) {
            return false;
        }

        // check quantize_linear zero_point
        auto zp_node_cast = std::dynamic_pointer_cast<Constant>(opsMap.at(dq_zp_label).get_node_shared_ptr());
        float zp;
        if (!zp_node_cast || !ov::op::util::get_single_value(zp_node_cast, zp)) {
            return false;
        }

        // prepare levels
        const auto& clamp_node_cast = std::dynamic_pointer_cast<Clamp>(opsMap.at(q_clamp_label).get_node_shared_ptr());
        if (!clamp_node_cast) {
            return false;
        }
        const auto high_range = static_cast<int>(clamp_node_cast->get_max());
        const auto low_range = static_cast<int>(clamp_node_cast->get_min());
        const auto levels = high_range - low_range + 1;

        // get the scale
        const auto& scale_node_cast = std::dynamic_pointer_cast<Constant>(
            opsMap.at(q_real_scale_label).get_node_shared_ptr()->get_input_node_shared_ptr(0));
        float scale;
        if (!scale_node_cast || !ov::op::util::get_single_value(scale_node_cast, scale)) {
            return false;
        }
        // The PaddleSlim scale value is not equal to scale definition in OpenVINO.
        // scale_ov = scale_pdpd / half_range.
        const auto real_scale = scale / high_range;

        // calculate the input_low/input_high/output_low/output_high
        // In order to reduce the imported nodes, try to achieve the value from the Constant.
        // The formula:
        // i8: which is used in PDPD
        //      low = (-128 - zero_point) * scale
        //      high = (127 - zero_point) * scale
        // u8: which is not used in PDPD
        //      low = (0 - zero_point) * scale
        //      high = (255 - zero_point) * scale
        const auto limit_low = std::make_shared<Constant>(element::f32, Shape{1}, (low_range - zp) * real_scale);
        const auto limit_high = std::make_shared<Constant>(element::f32, Shape{1}, (high_range - zp) * real_scale);

        auto fake_node =
            std::make_shared<FakeQuantize>(input_item, limit_low, limit_high, limit_low, limit_high, levels);
        fake_node->set_friendly_name(output_node->get_friendly_name());
        replace_node(output_node, fake_node);
        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(output_label, "TransformFakeQuantize");
    this->register_matcher(m, callback);
}
