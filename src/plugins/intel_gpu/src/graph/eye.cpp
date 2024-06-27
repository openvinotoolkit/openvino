// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <eye_inst.h>
#include "openvino/op/eye.hpp"
#include "eye_shape_inference.hpp"
#include <json_object.h>

#include <sstream>

#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(eye)

eye_inst::typed_primitive_inst(network& network, eye_node const& node) : parent(network, node) {}

template<typename ShapeType>
std::vector<layout> eye_inst::calc_output_layouts(eye_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto primitive = impl_param.typed_desc<eye>();
    const auto& input_layout = impl_param.get_input_layout();
    TensorsContainer const_data(&impl_param.get_stream());

    auto& memory_deps = impl_param.memory_deps;
    for (size_t i = 0; i < 4; i++) {
        if (memory_deps.count(i) > 0)
            const_data.emplace(i, memory_deps.at(i));
    }

    auto ta = cldnn::make_tensor_accessor(const_data);

    std::vector<ShapeType> input_shapes;
    for (size_t i = 0; i < impl_param.input_layouts.size(); i++) {
        input_shapes.push_back(impl_param.get_input_layout(i).get<ShapeType>());
    }

    ov::op::v9::Eye op;
    auto out_shapes = ov::op::v9::shape_infer(&op, input_shapes, ta);
    return { layout{ out_shapes[0], *(primitive->output_data_types[0]), input_layout.format} };
}

template std::vector<layout> eye_inst::calc_output_layouts<ov::PartialShape>(eye_node const& node, const kernel_impl_params& impl_param);

layout eye_inst::calc_output_layout(eye_node const& node, const kernel_impl_params&) {
    auto primitive = node.get_primitive();
    return {*(primitive->output_data_types[0]), node.get_input_layout().format, primitive->output_shape};
}

std::string eye_inst::to_string(eye_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite eye_info;
    eye_info.add("rows id", node.get_dependency(0).id());
    eye_info.add("cols id", node.get_dependency(1).id());
    eye_info.add("diagInd id", node.get_dependency(2).id());
    if (node.get_dependencies().size() == 4)
        eye_info.add("batchShape id", node.get_dependency(3).id());
    node_info->add("slice info", eye_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
