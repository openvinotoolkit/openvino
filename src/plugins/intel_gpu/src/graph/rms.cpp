// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "dynamic_quantize_inst.h"
#include "json_object.h"
#include "ov_ops/dynamic_quantize.hpp"
#include "primitive_type_base.h"
#include "rms_inst.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(rms);

template <typename ShapeType>
std::vector<layout> rms_inst::calc_output_layouts(rms_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<rms>();
    auto input_layout = impl_param.get_input_layout();
    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);

    if (impl_param.has_fused_primitives()) {
        const auto& fused_prims = node.get_fused_primitives();
        auto dq_it = std::find_if(fused_prims.begin(), fused_prims.end(), [](const cldnn::fused_primitive_desc& f) {
            return f.is_type<dynamic_quantize>();
        });
        if (dq_it != fused_prims.end()) {
            OPENVINO_ASSERT(std::addressof(*dq_it) == std::addressof(fused_prims.back()), "Dynamic quantize should be the last fused operation!");
            ov::op::internal::DynamicQuantize dq_op;
            dq_op.set_attrs(fused_prims.back().typed_desc<dynamic_quantize>()->attrs);
            const auto output_shapes = ov::op::internal::DynamicQuantize::shape_infer(&dq_op, {input_layout.get_partial_shape()});
            auto dq_attrs = fused_prims.back().typed_desc<dynamic_quantize>()->attrs;
            return {layout{output_shapes[0], dq_attrs.quantization_dt, input_layout.format}, layout{output_shapes[1], dq_attrs.scale_dt, input_layout.format}};
        } else {
            output_type = impl_param.get_output_element_type();
        }
    }

    return { layout(input_layout.get_partial_shape(), output_type, input_layout.format) };
}

template std::vector<layout> rms_inst::calc_output_layouts<ov::PartialShape>(
    rms_node const& node,
    const kernel_impl_params& impl_param);

layout rms_inst::calc_output_layout(rms_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<rms>();
    auto input_layout = impl_param.get_input_layout();
    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    return layout(output_type, input_layout.format, input_layout.get_tensor());
}

std::string rms_inst::to_string(rms_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite rms_info;
    rms_info.add("input_id", node.input(0).id());
    rms_info.add("epsilon", desc->epsilon);

    node_info->add("rms_info", rms_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

rms_inst::typed_primitive_inst(network& network, rms_node const& node) : parent(network, node) {}

}  // namespace cldnn
