// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_per_layer_scaling.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "openvino/op/multiply.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

FullyConnectedPerLayerScaling::FullyConnectedPerLayerScaling(float scale_factor) {
    using namespace ov::pass::pattern;

    auto data_m = any_input();
    auto weights_m = any_input();
    auto bias_m = any_input();
    auto fc_compressed_wo_zp_m = wrap_type<op::FullyConnectedCompressed>({data_m, weights_m, bias_m, any_input()}, consumers_count(1));
    auto fc_compressed_w_zp_m = wrap_type<op::FullyConnectedCompressed>({data_m, weights_m, bias_m, any_input(), any_input()}, consumers_count(1));
    auto fc_compressed_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fc_compressed_wo_zp_m, fc_compressed_w_zp_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        if (scale_factor == 0.f || scale_factor == 1.f)
            return false;
        auto fc = ov::as_type_ptr<op::FullyConnectedCompressed>(m.get_match_root());
        if (!fc || transformation_callback(fc))
            return false;

        const auto& pattern_map = m.get_pattern_value_map();
        const auto& data = pattern_map.at(data_m).get_node_shared_ptr();
        const auto& bias = pattern_map.at(bias_m).get_node_shared_ptr();

        ov::Shape scale_const_shape = {1};
        std::vector<float> scale_down_value = {(1.f / scale_factor)};
        std::vector<float> scale_up_value = {scale_factor};
        std::shared_ptr<ov::Node> scale_down_const_f16 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, scale_down_value);
        std::shared_ptr<ov::Node> scale_down_const_f32 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, scale_const_shape, scale_down_value);
        std::shared_ptr<ov::Node> scale_up_const_f16 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, scale_up_value);
        std::shared_ptr<ov::Node> scale_up_const_f32 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, scale_const_shape, scale_up_value);

        std::shared_ptr<ov::Node> scale_down_const = (data->get_element_type() == ov::element::f16) ? scale_down_const_f16 : scale_down_const_f32;
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(data, scale_down_const);
        scale_down->set_friendly_name(fc->get_friendly_name() + "_scale_down");
        ov::copy_runtime_info(fc, scale_down);
        fc->input(0).replace_source_output(scale_down);

        // If FC has bias as input, scaling must be applied to bias as well
        if (!ov::as_type_ptr<op::Placeholder>(bias)) {
            std::shared_ptr<ov::Node> bias_scale_down_const = (bias->get_element_type() == ov::element::f16) ? scale_down_const_f16 : scale_down_const_f32;
            auto bias_scale_down = std::make_shared<ov::op::v1::Multiply>(bias, bias_scale_down_const);
            bias_scale_down->set_friendly_name(fc->get_friendly_name() + "_bias_scale_down");
            ov::copy_runtime_info(fc, bias_scale_down);
            fc->input(2).replace_source_output(bias_scale_down);
        }

        auto target_inputs = fc->get_output_target_inputs(0);
        std::shared_ptr<ov::Node> scale_up_const = (fc->get_element_type() == ov::element::f16) ? scale_up_const_f16 : scale_up_const_f32;
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(fc, scale_up_const);
        scale_up->set_friendly_name(fc->get_friendly_name() + "_scale_up");
        ov::copy_runtime_info(fc, scale_up);
        for (auto& in : target_inputs) {
            in.replace_source_output(scale_up);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_compressed_m, "FullyConnectedPerLayerScaling");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
