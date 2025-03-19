// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_fc_to_quantized_legacy.hpp"

#include <memory>

#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_quantized_legacy.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertFCToFCQuantizedLegacy::ConvertFCToFCQuantizedLegacy() {
    using namespace ov::pass::pattern;

    std::vector<element::Type> activation_types{ov::element::u8, ov::element::i8};
    std::vector<element::Type> weights_types{ov::element::i8};

    auto activations_m = pattern::any_input(ov::pass::pattern::type_matches_any(activation_types));
    auto weights_m = pattern::any_input();
    auto bias_m = pattern::any_input();

    auto fully_connected_m = wrap_type<ov::op::internal::FullyConnected>({activations_m, weights_m, bias_m});
    auto dequantization_scales_m = wrap_type<ov::op::v0::Constant>();
    auto multiply_m = wrap_type<ov::op::v1::Multiply>({fully_connected_m, dequantization_scales_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto fc_output = pattern_map.at(fully_connected_m);
        auto activations = pattern_map.at(activations_m);
        auto weights = pattern_map.at(weights_m);
        auto bias = pattern_map.at(bias_m);
        auto multiply = pattern_map.at(multiply_m);
        auto dequantization_scales = pattern_map.at(dequantization_scales_m);
        const auto& fc_output_shape = fc_output.get_partial_shape();
        const auto& multiply_output_shape = multiply.get_partial_shape();

        if (*fc_output_shape.rbegin() != *multiply_output_shape.rbegin() ||
            !ov::op::util::is_on_constant_path(weights)) {
            return false;
        }

        auto fc_node =
            ov::as_type_ptr<ov::op::internal::FullyConnected>(pattern_map.at(fully_connected_m).get_node_shared_ptr());

        ov::NodeVector new_ops;
        auto zp = std::make_shared<ov::op::v0::Constant>(element::dynamic, Shape{0});
        new_ops.push_back(zp);

        auto fc_quantized =
            std::make_shared<ov::op::internal::FullyConnectedQuantizedLegacy>(activations,
                                                                              weights,
                                                                              bias,
                                                                              dequantization_scales,
                                                                              zp,
                                                                              fc_node->get_output_type());
        new_ops.push_back(fc_quantized);

        const auto& multiply_node = multiply.get_node_shared_ptr();
        fc_quantized->set_friendly_name(multiply_node->get_friendly_name());

        ov::copy_runtime_info({multiply_node, fc_node}, new_ops);
        ov::replace_node(multiply_node, fc_quantized);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(multiply_m, "ConvertFullyConnectedToFullyConnectedQuantized");
    this->register_matcher(m, callback);
}
