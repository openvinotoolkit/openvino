// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fake_convert_inst.h"
#include "fake_convert_shape_inference.hpp"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(fake_convert)

layout fake_convert_inst::calc_output_layout(fake_convert_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
std::vector<layout> fake_convert_inst::calc_output_layouts(fake_convert_node const& node, kernel_impl_params const& impl_param) {
    const auto& input_layout = impl_param.get_input_layout(0);
    auto output_type = ov::element::Type(input_layout.data_type);

    OPENVINO_ASSERT(ov::element::Type::merge(output_type, output_type, ov::element::Type(impl_param.get_input_layout(1).data_type)),
        "Mixed input types are not supported.");

    if (impl_param.input_layouts.size() == 3) {
        OPENVINO_ASSERT(ov::element::Type::merge(output_type, output_type, ov::element::Type(impl_param.get_input_layout(2).data_type)),
            "Mixed input types are not supported.");
    }

    switch (output_type) {
    case ov::element::bf16:
    case ov::element::f16:
    case ov::element::f32:
        break;
    default:
        OPENVINO_THROW("The output data type should be a bf16, f16, f32 but got: ", output_type);
    }

    return { layout{input_layout.get_partial_shape(), output_type, input_layout.format} };
}

template std::vector<layout> fake_convert_inst::calc_output_layouts<ov::PartialShape>(fake_convert_node const& node, const kernel_impl_params& impl_param);

std::string fake_convert_inst::to_string(fake_convert_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();
    auto& scale = node.scale();

    std::stringstream primitive_description;

    json_composite fake_convert_info;
    fake_convert_info.add("input id", input.id());
    fake_convert_info.add("scale id", scale.id());
    if (node.has_shift()) {
        fake_convert_info.add("shift id", node.shift().id());
    }
    fake_convert_info.add("destination_type", node.get_destination_type().get_type_name());

    node_info->add("fake_convert info", fake_convert_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

fake_convert_inst::typed_primitive_inst(network& network, fake_convert_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
