// Copyright (C) 2018-2025 Intel Corporation
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

layout scatter_elements_update_inst::calc_output_layout(scatter_elements_update_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<scatter_elements_update>();

    const int32_t axis = desc->axis;
    const size_t input_number_of_dims = impl_param.get_input_layout().get_partial_shape().size();

    auto input_layout = impl_param.get_input_layout();

    auto output_shape = input_layout.get_partial_shape();
    auto input_format = input_layout.format;
    auto output_type = input_layout.data_type;

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    if (static_cast<size_t>(axis) < 0 || static_cast<size_t>(axis) >= input_number_of_dims)
        CLDNN_ERROR_MESSAGE(desc->id, "Incorrect axis value for ScatterElementsUpdate: Axis must be positive and less than the input tensor dimension.");

    return layout{output_shape, output_type, input_format};
}

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

scatter_elements_update_inst::typed_primitive_inst(network& network, scatter_elements_update_node const& node) : parent(network, node) {
    update_output_memory();
}

void scatter_elements_update_inst::on_execute() {
    update_output_memory();
}

void scatter_elements_update_inst::update_output_memory() {
    if (!can_be_optimized() || _impl_params->is_dynamic())
        return;

    if (_outputs.size() > 0 && static_cast<bool>(_outputs[0])
        && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    if (_node != nullptr)
        build_deps();

    // Can_be_optimized nodes are allocating from memory_pool too. In this case,
    // we need release the legacy output memory from memory pool explicitly.
    if (static_cast<bool>(_outputs[0]) &&
        _node->get_program().get_config().get_enable_memory_pool()) {
        _network.get_memory_pool().release_memory(_outputs[0].get(), _node->get_unique_id(), _node->id(), _network.get_id());
    }
    _outputs = {_network.get_engine().reinterpret_buffer(input_memory(), _impl_params->get_output_layout())};
    _mem_allocated = false;
}
}  // namespace cldnn
