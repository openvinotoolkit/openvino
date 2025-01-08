// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/fc_dynamic_quantization.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "ov_ops/fully_connected_quantized.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {

using QuantizationType = ov::op::internal::DynamicQuantize::QuantizationType;

FullyConnectedDynamicQuantization::FullyConnectedDynamicQuantization(uint64_t group_size, const ov::element::Type& quantization_dt)
    : ov::pass::MatcherPass() {
    using namespace ov::pass::pattern;

    std::vector<element::Type> weights_types{ov::element::i8, ov::element::u8, ov::element::i4, ov::element::u4};
    std::vector<element::Type> zero_points_types{ov::element::i8, ov::element::u8, ov::element::i4, ov::element::u4, ov::element::undefined};

    auto activations_m = pattern::any_input();
    auto weights_m = pattern::any_input(ov::pass::pattern::type_matches_any(weights_types));
    auto bias_m = pattern::any_input();
    auto weights_scales_m = pattern::any_input();
    auto weights_zero_points_m = pattern::any_input(ov::pass::pattern::type_matches_any(zero_points_types));

    auto fc_compressed_m = wrap_type<ov::op::internal::FullyConnectedCompressed>({activations_m, weights_m, bias_m, weights_scales_m, weights_zero_points_m});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(activations_m));
        OPENVINO_ASSERT(pattern_map.count(weights_m));
        OPENVINO_ASSERT(pattern_map.count(bias_m));
        OPENVINO_ASSERT(pattern_map.count(weights_scales_m));
        OPENVINO_ASSERT(pattern_map.count(weights_zero_points_m));
        auto fc_compressed = std::dynamic_pointer_cast<ov::op::internal::FullyConnectedCompressed>(
            pattern_map.at(fc_compressed_m).get_node_shared_ptr());
        if (!fc_compressed || transformation_callback(fc_compressed)) {
            return false;
        }

        auto weight_shape = fc_compressed->get_input_partial_shape(1);
        const size_t IC = weight_shape[weight_shape.size() - 1].get_length();
        if (group_size != UINT64_MAX && (group_size == 0 || (IC % group_size != 0 && IC > group_size))) {
            return false;
        }

        auto rank = fc_compressed->get_input_partial_shape(0).size();
        std::vector<uint64_t> shape_group_size(rank, 1);
        shape_group_size.back() = group_size;

        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_dt = quantization_dt;
        dq_config.quantization_type = quantization_dt.is_signed() ? QuantizationType::Symmetric : QuantizationType::Asymmetric;
        dq_config.scale_dt = fc_compressed->get_input_element_type(0);
        dq_config.group_sizes = shape_group_size;
        dq_config.zp_dt = quantization_dt;

        auto dynamic_quantize = std::make_shared<ov::op::internal::DynamicQuantize>(pattern_map.at(activations_m).get_node_shared_ptr(), dq_config);
        // auto optional_w_zp = m_fc->get_input_size() > 4 ? m_fc->get_input_node_shared_ptr(4) : std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto input_zp = dq_config.quantization_type == QuantizationType::Symmetric ? std::make_shared<ov::op::v0::Constant>(element::undefined, Shape{0}) : dynamic_quantize->output(2);

        auto output_scales = std::make_shared<ov::op::v0::Constant>(element::undefined, Shape{0});
        auto output_zero_points = std::make_shared<ov::op::v0::Constant>(element::undefined, Shape{0});

        auto output_type = fc_compressed->get_output_type();
        if (output_type == ov::element::undefined)
            output_type = fc_compressed->get_input_element_type(0);

        auto fc_quantized = std::make_shared<ov::op::internal::FullyConnectedQuantized>(dynamic_quantize->output(0),
                                                                     pattern_map.at(weights_m).get_node_shared_ptr(),
                                                                     pattern_map.at(bias_m).get_node_shared_ptr(),
                                                                     pattern_map.at(weights_scales_m).get_node_shared_ptr(),
                                                                     pattern_map.at(weights_zero_points_m).get_node_shared_ptr(),
                                                                     dynamic_quantize->output(1),
                                                                     input_zp,
                                                                     output_scales,
                                                                     output_zero_points,
                                                                     output_type);

        ov::replace_node(fc_compressed, fc_quantized);

        fc_quantized->set_friendly_name(fc_compressed->get_friendly_name());
        ov::copy_runtime_info(fc_compressed, fc_quantized);

        // if (transformation_callback(m.get_match_root())) {
        //     return false;
        // }
        // const auto& pattern_map = m.get_pattern_value_map();
        // const auto& m_data = pattern_map.at(data).get_node_shared_ptr();

        // auto m_fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(m.get_match_root());

        // auto weight_shape = m_fc->get_input_partial_shape(1);
        // const size_t innermost_size = weight_shape[weight_shape.size() - 1].get_length();
        // if (group_size != UINT64_MAX &&
        //     (group_size == 0 || (innermost_size % group_size != 0 && innermost_size > group_size))) {
        //     GPU_DEBUG_TRACE << "Dynamic quantization: shape is not aligned with group size " << innermost_size << " / " << group_size << std::endl;
        //     return false;
        // }

        // auto rank = m_fc->get_input_partial_shape(0).size();
        // std::vector<uint64_t> shape_group_size(rank, 1);
        // shape_group_size.back() = group_size;

        // ov::op::internal::DynamicQuantize::Attributes config;
        // config.quantization_dt = element::i8;
        // config.quantization_type = QuantizationType::Symmetric;
        // config.scale_dt = element::f16;
        // config.group_sizes = shape_group_size;

        // GPU_DEBUG_IF(debug_config->dynamic_quantize_asym) {
        //     config.quantization_type = QuantizationType::Asymmetric;
        //     config.quantization_dt = element::u8;
        //     config.zp_dt = element::u8; // it supports u8 only now
        // }

        // auto dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(m_data, config);
        // auto optional_w_zp = m_fc->get_input_size() > 4 ? m_fc->get_input_node_shared_ptr(4) : std::make_shared<ov::intel_gpu::op::Placeholder>();
        // auto optional_a_zp = config.quantization_type == QuantizationType::Symmetric ?
        //                         std::make_shared<ov::intel_gpu::op::Placeholder>() : dyn_quan->output(2);

        // auto output_type = m_fc->get_output_type();
        // if (output_type == ov::element::undefined)
        //     output_type = m_fc->get_input_element_type(0);

        // auto new_fc = std::make_shared<op::FullyConnectedCompressed>(dyn_quan->output(0),
        //                                                              m_fc->get_input_node_shared_ptr(1),
        //                                                              m_fc->get_input_node_shared_ptr(2),
        //                                                              m_fc->get_input_node_shared_ptr(3),
        //                                                              optional_w_zp,
        //                                                              dyn_quan->output(1),
        //                                                              optional_a_zp,
        //                                                              output_type);

        // ov::replace_node(m_fc, new_fc);

        // new_fc->set_friendly_name(m_fc->get_friendly_name());
        // ov::copy_runtime_info(m_fc, new_fc);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_compressed_m, "FullyConnectedDynamicQuantization");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
