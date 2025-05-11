// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shuffle_channels_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(shuffle_channels)

layout shuffle_channels_inst::calc_output_layout(shuffle_channels_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<shuffle_channels>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;

    const int32_t number_of_dims = 4;
    const int32_t group = desc->group;
    int32_t axis = desc->axis;

    if (axis < 0)
        axis += number_of_dims;

    if (axis < 0 || axis >= number_of_dims)
        CLDNN_ERROR_MESSAGE(desc->id, "Incorrect axis value! Actual axis is" + std::to_string(group));

    if (group < 1)
        CLDNN_ERROR_MESSAGE(
            desc->id,
            "Invalid group size value (should equal at least one). Actual block size is" + std::to_string(group));

    if (input_layout.get_tensor().sizes(format::bfyx)[axis] % group != 0)
        CLDNN_ERROR_MESSAGE(
            desc->id,
            "Group parameter must evenly divide the channel dimension. Actual group size is " + std::to_string(group));

    return layout{input_layout.data_type, input_format, input_layout.get_tensor()};
}

std::string shuffle_channels_inst::to_string(shuffle_channels_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite shuffle_channels_info;
    shuffle_channels_info.add("input id", input.id());
    shuffle_channels_info.add("groups number", desc->group);
    shuffle_channels_info.add("axis", desc->axis);

    node_info->add("shuffle_channels info", shuffle_channels_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

shuffle_channels_inst::typed_primitive_inst(network& network, shuffle_channels_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
