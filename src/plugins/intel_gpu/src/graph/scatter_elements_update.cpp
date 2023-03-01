// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/error_handler.hpp"
#include "scatter_elements_update_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(scatter_elements_update)

layout scatter_elements_update_inst::calc_output_layout(scatter_elements_update_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<scatter_elements_update>();

    const int32_t axis = desc->axis;
    const size_t input_number_of_dims = impl_param.get_input_layout().get_tensor().sizes().size();

    auto input_layout = impl_param.get_input_layout();

    auto output_shape = input_layout.get_tensor();
    auto input_format = input_layout.format;
    auto output_type = input_layout.data_type;

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    if (static_cast<size_t>(axis) < 0 || static_cast<size_t>(axis) >= input_number_of_dims)
        CLDNN_ERROR_MESSAGE(desc->id, "Incorrect axis value for ScatterElementsUpdate: Axis must be positive and less than the input tensor dimension.");

    return layout{output_type, input_format, output_shape};
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

scatter_elements_update_inst::typed_primitive_inst(network& network, scatter_elements_update_node const& node) : parent(network, node) {}

}  // namespace cldnn
