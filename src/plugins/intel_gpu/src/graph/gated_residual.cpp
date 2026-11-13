// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gated_residual_inst.h"
#include "intel_gpu/primitives/gated_residual_caps.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {

bool gated_residual_caps::uses_dpas_kernels(const device_info& info, const ov::intel_gpu::ExecutionConfig& config) {
    const bool force_scalar = GPU_DEBUG_VALUE_OR(config.get_gr_kernel_selection(), std::string("auto")) == "scalar";
    return !force_scalar && is_device_supported(info) && info.arch >= gpu_arch::xe2 && info.supports_immad;
}

GPU_DEFINE_PRIMITIVE_TYPE_ID(gated_residual)

layout gated_residual_inst::calc_output_layout(const gated_residual_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> gated_residual_inst::calc_output_layouts(const gated_residual_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<gated_residual>();
    const auto residual_layout = impl_param.get_input_layout(0);
    const auto out_dt =
        desc->output_data_types.empty() || !desc->output_data_types[0].has_value() ? residual_layout.data_type : desc->output_data_types[0].value();

    if (desc->mode == gated_residual::Mode::READ) {
        OPENVINO_ASSERT(residual_layout.get_partial_shape().rank().is_static() && residual_layout.get_partial_shape().size() >= 2,
                        "[GPU] gated_residual requires a static-rank residual input of rank >= 2");
    }

    const auto shapes = ov::intel_gpu::op::GatedResidual::infer_output_shapes(desc->mode,
                                                                              desc->hc_count,
                                                                              desc->hidden_size,
                                                                              desc->produce_write_gate,
                                                                              residual_layout.get_partial_shape());
    std::vector<layout> output_layouts;
    output_layouts.emplace_back(shapes[0], out_dt, residual_layout.format);
    if (shapes.size() > 1) {
        const auto gate_dt = desc->output_data_types.size() > 1 && desc->output_data_types[1].has_value() ? desc->output_data_types[1].value() : out_dt;
        output_layouts.emplace_back(shapes[1], gate_dt, residual_layout.format);
    }

    return output_layouts;
}

template std::vector<layout> gated_residual_inst::calc_output_layouts<ov::PartialShape>(const gated_residual_node& node, const kernel_impl_params& impl_param);

std::string gated_residual_inst::to_string(const gated_residual_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    info.add("mode", static_cast<int>(desc->mode));
    info.add("hc_count", desc->hc_count);
    info.add("hidden_size", desc->hidden_size);
    info.add("lowrank", desc->lowrank);
    info.add("epsilon", desc->epsilon);
    info.add("produce_write_gate", desc->produce_write_gate);

    node_info->add("gated_residual_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gated_residual_inst::typed_primitive_inst(network& network, const gated_residual_node& node) : parent(network, node) {}

}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gated_residual)
