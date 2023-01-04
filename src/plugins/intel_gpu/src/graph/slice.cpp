// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <slice_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(slice)

slice_inst::typed_primitive_inst(network& network, slice_node const& node)
    : parent(network, node) {}

layout slice_inst::calc_output_layout(slice_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<slice>();
    auto input_layout = impl_param.get_input_layout();
    return {input_layout.data_type, input_layout.format, primitive->output_shape};
}

std::string slice_inst::to_string(slice_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite slice_info;
    slice_info.add("input id", node.input().id());
    slice_info.add("begin_param id", node.get_dependency(1).id());
    slice_info.add("end_param id", node.get_dependency(2).id());
    slice_info.add("step_param id", node.get_dependency(3).id());
    slice_info.add("axis_param id", node.get_dependency(4).id());
    node_info->add("slice info", slice_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

} // namespace cldnn
