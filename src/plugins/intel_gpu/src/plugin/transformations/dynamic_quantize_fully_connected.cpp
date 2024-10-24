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

namespace ov {
namespace intel_gpu {

DynamicQuantizeFullyConnected::DynamicQuantizeFullyConnected(uint64_t group_size) {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    using namespace ov::pass::pattern;

    // per-token quantization is supported
    if (group_size != UINT64_MAX) {
        GPU_DEBUG_TRACE << "Dynamic quantization is disabled " << group_size << std::endl;
        return;
    }
    auto is_dynamic = [](const ov::Output<ov::Node>& output) -> bool {
        bool is_dynamic = output.get_node_shared_ptr()->get_output_partial_shape(0).is_dynamic();
        size_t num_inputs = output.get_node_shared_ptr()->get_input_size();
        for (size_t idx = 0; idx < num_inputs; idx++) {
            is_dynamic |= output.get_node_shared_ptr()->get_input_partial_shape(idx).is_dynamic();
        }
        return is_dynamic;
    };

    auto data = any_input();
    auto fully_connected_compressed3 = wrap_type<op::FullyConnectedCompressed>({data, any_input(), any_input(), any_input()}, is_dynamic);
    auto fully_connected_compressed4 = wrap_type<op::FullyConnectedCompressed>({data, any_input(), any_input(), any_input(), any_input()}, is_dynamic);
    auto fully_connected_compressed = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fully_connected_compressed3, fully_connected_compressed4});


    ov::matcher_pass_callback callback = [=](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& m_data = pattern_map.at(data).get_node_shared_ptr();

        auto m_fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(m.get_match_root());

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

        ov::op::internal::QuantizationConfig config;
        config.quantization_dt = element::i8;
        config.type = ov::op::internal::QuantizationConfig::QuantizationType::Symmetric;
        config.scale_dt = element::f16;
        config.group_sizes = shape_group_size;

        auto dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(m_data, config);
        auto optional_w_zp = m_fc->get_input_size() > 4 ? m_fc->get_input_node_shared_ptr(4) : std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto output_type = m_fc->get_output_type();
        if (output_type == ov::element::undefined)
            output_type = m_fc->get_input_element_type(0);

        auto new_fc = std::make_shared<op::FullyConnectedCompressed>(dyn_quan->output(0),
                                                                     m_fc->get_input_node_shared_ptr(1),
                                                                     m_fc->get_input_node_shared_ptr(2),
                                                                     m_fc->get_input_node_shared_ptr(3),
                                                                     optional_w_zp,
                                                                     dyn_quan->output(1),
                                                                     output_type);

        ov::replace_node(m_fc, new_fc);

        new_fc->set_friendly_name(m_fc->get_friendly_name());
        ov::copy_runtime_info(m_fc, new_fc);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_compressed, "DynamicQuantizeFullyConnected");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
