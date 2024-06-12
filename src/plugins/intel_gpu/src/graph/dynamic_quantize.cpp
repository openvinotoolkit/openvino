// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/dynamic_quantize.hpp"
#include "dynamic_quantize_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(dynamic_quantize);

layout dynamic_quantize_inst::calc_output_layout(dynamic_quantize_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    auto input_layout = impl_param.get_input_layout();
    auto output_type = data_types::i8;
    auto output_format = input_layout.format;

    return layout(output_type, output_format, input_layout.get_tensor());
}

template<typename ShapeType>
std::vector<layout> dynamic_quantize_inst::calc_output_layouts(dynamic_quantize_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    auto input_layout = impl_param.get_input_layout();
    auto output_type = data_types::i8;
    auto output_format = input_layout.format;

    ov::intel_gpu::op::DynamicQuantize op;

    std::vector<ShapeType> input_shapes = {
        impl_param.get_input_layout(0).get<ShapeType>(),
    };

    std::vector<ShapeType> output_shapes = shape_infer(&op, input_shapes);

    return { layout(output_shapes[0], output_type, output_format) };
}

template std::vector<layout> dynamic_quantize_inst::calc_output_layouts<ov::PartialShape>(dynamic_quantize_node const& node,
                                                                                const kernel_impl_params& impl_param);

std::string dynamic_quantize_inst::to_string(dynamic_quantize_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

dynamic_quantize_inst::typed_primitive_inst(network& network, dynamic_quantize_node const& node) : parent(network, node) {}

}  // namespace cldnn
