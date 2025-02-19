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

namespace ov::intel_gpu {

DynamicQuantizeFullyConnected::DynamicQuantizeFullyConnected(uint64_t group_size, bool asymmetric)
    : ov::pass::MatcherPass() {
    using namespace ov::pass::pattern;
    using QuantizationType = ov::op::internal::DynamicQuantize::QuantizationType;

    auto data = any_input();
    auto fully_connected_compressed3 = wrap_type<op::FullyConnectedCompressed>({data, any_input(), any_input(), any_input()});
    auto fully_connected_compressed4 = wrap_type<op::FullyConnectedCompressed>({data, any_input(), any_input(), any_input(), any_input()});
    auto fully_connected_compressed = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected_compressed3, fully_connected_compressed4});


    ov::matcher_pass_callback callback = [=](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& m_data = pattern_map.at(data).get_node_shared_ptr();

        auto m_fc = ov::as_type_ptr<op::FullyConnectedCompressed>(m.get_match_root());

        auto weight_shape = m_fc->get_input_partial_shape(1);
        const size_t innermost_size = weight_shape[weight_shape.size() - 1].get_length();
        if (group_size != UINT64_MAX &&
            (group_size == 0 || (innermost_size % group_size != 0 && innermost_size > group_size))) {
            GPU_DEBUG_TRACE << "Dynamic quantization: shape is not aligned with group size " << innermost_size << " / " << group_size << std::endl;
            return false;
        }

        auto rank = m_fc->get_input_partial_shape(0).size();
        std::vector<uint64_t> shape_group_size(rank, 1);
        shape_group_size.back() = group_size;

        ov::op::internal::DynamicQuantize::Attributes config;
        config.quantization_dt = element::i8;
        config.quantization_type = QuantizationType::Symmetric;
        config.scale_dt = element::f16;
        config.group_sizes = shape_group_size;

        if (asymmetric && group_size == UINT64_MAX) {
            config.quantization_type = QuantizationType::Asymmetric;
            config.quantization_dt = element::u8;
            config.zp_dt = element::u8; // it supports u8 only now
        }

        auto dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(m_data, config);
        auto optional_w_zp = m_fc->get_input_size() > 4 ? m_fc->get_input_node_shared_ptr(4) : std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto optional_a_zp = config.quantization_type == QuantizationType::Symmetric ?
                                std::make_shared<ov::intel_gpu::op::Placeholder>() : dyn_quan->output(2);

        auto output_type = m_fc->get_output_type();
        if (output_type.is_dynamic())
            output_type = m_fc->get_input_element_type(0);

        auto new_fc = std::make_shared<op::FullyConnectedCompressed>(dyn_quan->output(0),
                                                                     m_fc->get_input_node_shared_ptr(1),
                                                                     m_fc->get_input_node_shared_ptr(2),
                                                                     m_fc->get_input_node_shared_ptr(3),
                                                                     optional_w_zp,
                                                                     dyn_quan->output(1),
                                                                     optional_a_zp,
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
