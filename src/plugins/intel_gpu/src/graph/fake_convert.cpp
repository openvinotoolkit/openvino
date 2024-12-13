// Copyright (C) 2018-2024 Intel Corporation
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
    auto desc = impl_param.typed_desc<fake_convert>();
    auto input_layout = impl_param.get_input_layout(0);
    auto scale_layout = impl_param.get_input_layout(1);
    auto output_type    = input_layout.data_type;
    auto output_format  = input_layout.format;

    ov::op::v13::FakeConvert op;

    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        scale_layout.get<ShapeType>()
    };
    if (impl_param.input_layouts.size() == 3) {
        auto shift_layout = impl_param.get_input_layout(2);
        input_shapes.push_back(shift_layout.get<ShapeType>());
    }
    std::vector<ShapeType> output_shapes = ov::op::v13::shape_infer(&op, input_shapes);

    return { layout{output_shapes[0], output_type, output_format} };
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
    fake_convert_info.add("destination_type", node.get_destination_type());

    node_info->add("fake_convert info", fake_convert_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

fake_convert_inst::typed_primitive_inst(network& network, fake_convert_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
