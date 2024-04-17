// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/error_handler.hpp"
#include "scatter_elements_update_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

#include "scatter_elements_update_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(scatter_elements_update)

std::string scatter_elements_update_inst::to_string(scatter_elements_update_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite scatter_elements_update_info;
    scatter_elements_update_info.add("input id", input.id());
    scatter_elements_update_info.add("axis", desc->axis);

    node_info->add("scatter_elements_update info", scatter_elements_update_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scatter_elements_update_inst::typed_primitive_inst(network& network, scatter_elements_update_node const& node) : parent(network, node) {}
void scatter_elements_update_inst::on_execute() {
    auto input1_shape = _impl_params->input_layouts[1].get_partial_shape();
    auto input2_shape = _impl_params->input_layouts[2].get_partial_shape();

    if ((ov::shape_size(input1_shape.to_shape()) == 0) || (ov::shape_size(input2_shape.to_shape()) == 0))
        update_output_memory();
}

void scatter_elements_update_inst::update_output_memory() {
    if (_outputs.size() > 0 && static_cast<bool>(_outputs[0])
        && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    if (_node != nullptr)
        build_deps();

    _outputs = {_network.get_engine().reinterpret_buffer(input_memory(), _impl_params->get_output_layout())};
    _mem_allocated = false;
}
}  // namespace cldnn
