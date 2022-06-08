// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>

#include "bucketize_inst.hpp"
#include "json_object.h"
#include "primitive_type_base.h"

namespace cldnn {

primitive_type_id bucketize::type_id() {
    static primitive_type_base<bucketize> instance;
    return &instance;
}

layout bucketize_inst::calc_output_layout(const bucketize_node& node) {
    auto input_layout = node.input().get_output_layout();
    auto primitive = node.get_primitive();
    return {*primitive->output_data_type, input_layout.format, input_layout.size};
}

std::string bucketize_inst::to_string(const bucketize_node& node) {
    auto primitive = node.get_primitive();
    json_composite bucketize_info;
    bucketize_info.add("input id", node.input().id());
    bucketize_info.add("buckets id", node.buckets().id());
    bucketize_info.add("output_type", dt_to_str(*primitive->output_data_type));
    bucketize_info.add("with_right_bound", primitive->with_right_bound);

    auto node_info = node.desc_to_json();
    node_info->add("bucketize info", bucketize_info);

    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
