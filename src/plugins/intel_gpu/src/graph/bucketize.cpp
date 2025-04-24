// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>

#include "bucketize_inst.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(bucketize)

layout bucketize_inst::calc_output_layout(const bucketize_node& node, kernel_impl_params const& impl_param) {
    auto input_layout = impl_param.get_input_layout();
    auto primitive = impl_param.desc;
    return {*primitive->output_data_types[0], input_layout.format, input_layout.get_tensor()};
}

std::string bucketize_inst::to_string(const bucketize_node& node) {
    auto primitive = node.get_primitive();
    json_composite bucketize_info;
    bucketize_info.add("output_type", dt_to_str(*primitive->output_data_types[0]));
    bucketize_info.add("with_right_bound", primitive->with_right_bound);

    auto node_info = node.desc_to_json();
    node_info->add("bucketize info", bucketize_info);

    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
