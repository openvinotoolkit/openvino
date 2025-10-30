// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_fully_connected.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "ov_ops/dynamic_quantize.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov::intel_gpu {

// precomputed_reduction is providing partial reduction of activation from dynamic quantization into onednn for faster computation
// It is used for asymmetric weight.
DynamicQuantizeFullyConnected::DynamicQuantizeFullyConnected(uint64_t group_size,
                                                            bool asymmetric,
                                                            bool precomputed_reduction,
                                                            bool use_gs128_for_int8_per_token)
    : ov::pass::MatcherPass() {
    using namespace ov::pass::pattern;
    using QuantizationType = ov::op::internal::DynamicQuantize::QuantizationType;

    auto data = any_input();
    auto fully_connected_compressed3 = wrap_type<op::FullyConnectedCompressed>({data, any_input(), any_input(), any_input()});
    auto fully_connected_compressed4 = wrap_type<op::FullyConnectedCompressed>({data, any_input(), any_input(), any_input(), any_input()});
    auto fully_connected_compressed = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected_compressed3, fully_connected_compressed4});


    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        uint64_t adj_group_size = group_size; // If group_size is not supported, it can be adjusted to proper group size

        auto m_fc = ov::as_type_ptr<op::FullyConnectedCompressed>(m.get_match_root());

        auto weight_shape = m_fc->get_input_partial_shape(1);
        const size_t innermost_size = weight_shape[weight_shape.size() - 1].get_length();

        const bool has_wzp = m_fc->get_input_size() > 4;
        auto optional_w_zp = has_wzp ? m_fc->get_input_node_shared_ptr(4) : std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto rank = m_fc->get_input_partial_shape(0).size();
        ov::op::internal::DynamicQuantize::Attributes config;
        const bool has_static_wzp = m_fc->get_input_size() > 4 && optional_w_zp->get_output_partial_shape(0).rank().is_static();
        const bool is_wei_i8_u8 = cldnn::one_of(m_fc->get_input_element_type(1), {ov::element::i8, ov::element::u8});

        if (DynamicQuantizeFullyConnected::ShouldUseGs128(is_wei_i8_u8, use_gs128_for_int8_per_token, adj_group_size)) {
            adj_group_size = 128;
        }

        // Add precomputed_reduction connection, if possible
        if (precomputed_reduction && adj_group_size != UINT64_MAX && adj_group_size > 0 && has_static_wzp) {
            auto weight_zp_shape = m_fc->get_input_partial_shape(4);
            auto weight_scale_shape = m_fc->get_input_partial_shape(3);
            const size_t wei_zp_group_size = innermost_size / weight_zp_shape[weight_zp_shape.size() - 1].get_length();
            const size_t wei_scale_group_size = innermost_size / weight_scale_shape[weight_scale_shape.size() - 1].get_length();
            const size_t required_group_size = std::min(wei_zp_group_size, wei_scale_group_size);
            if (adj_group_size > required_group_size) {
                GPU_DEBUG_LOG << "Dynamic quantization: adjusting group_size " << adj_group_size
                               << " to wei_zp_group_size " << wei_zp_group_size << " and wei_scale_group_size " << wei_scale_group_size << std::endl;
                adj_group_size = required_group_size;
            }
            const bool is_per_token = adj_group_size == innermost_size;
            if (required_group_size % adj_group_size == 0 && !is_per_token) {
                config.precomputed_reduction_dt = element::i32; // it supports i32 only now
                config.precomputed_reduction = true;
            } else {
                GPU_DEBUG_LOG << "Dynamic quantization: precompute is turned off because group_size " << adj_group_size
                               << " is not aligned with required_group_size " << required_group_size << std::endl;
            }
        }

        if (adj_group_size != UINT64_MAX &&
            (adj_group_size == 0 || (innermost_size % adj_group_size != 0 && innermost_size > adj_group_size))) {
            GPU_DEBUG_TRACE << "Dynamic quantization: shape is not aligned with group size " << innermost_size << " / " << adj_group_size << std::endl;
            return false;
        }

        std::vector<uint64_t> shape_group_size(rank, 1);
        shape_group_size.back() = adj_group_size;

        config.quantization_dt = element::i8;
        config.quantization_type = QuantizationType::Symmetric;
        config.scale_dt = element::f16;
        config.group_sizes = shape_group_size;

        if (asymmetric && adj_group_size == UINT64_MAX) {
            config.quantization_type = QuantizationType::Asymmetric;
            config.quantization_dt = element::u8;
            config.zp_dt = element::u8; // it supports u8 only now
        }

        auto dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(m_fc->input_value(0), config);
        int dyn_quan_output_idx = 2;
        auto optional_a_zp = config.quantization_type == QuantizationType::Symmetric ?
                                std::make_shared<ov::intel_gpu::op::Placeholder>() : dyn_quan->output(dyn_quan_output_idx++);
        auto optional_precomputed_reduction = config.precomputed_reduction ?
                                 dyn_quan->output(dyn_quan_output_idx++) : std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto output_type = m_fc->get_output_type();
        if (output_type.is_dynamic())
            output_type = m_fc->get_input_element_type(0);

        auto new_fc = std::make_shared<op::FullyConnectedCompressed>(dyn_quan->output(0),
                                                                     m_fc->input_value(1),
                                                                     m_fc->input_value(2),
                                                                     m_fc->input_value(3),
                                                                     optional_w_zp->output(0),
                                                                     dyn_quan->output(1),
                                                                     optional_a_zp,
                                                                     optional_precomputed_reduction,
                                                                     output_type);

        ov::replace_node(m_fc, new_fc);

        new_fc->set_friendly_name(m_fc->get_friendly_name());
        ov::copy_runtime_info(m_fc, new_fc);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_compressed, "DynamicQuantizeFullyConnected");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
