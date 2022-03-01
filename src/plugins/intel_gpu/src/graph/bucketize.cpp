// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "bucketize_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"

namespace cldnn {
primitive_type_id bucketize::type_id() {
    static primitive_type_base<bucketize> instance;
    return &instance;
}

layout bucketize_inst::calc_output_layout(bucketize_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for bucketize_node!");

    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;

    data_types output_type = desc->output_type;

    if (input_layout.format.dimension() == 5) {
        return layout{output_type,
                      input_format,
                      tensor(TensorValue(input_layout.size.batch[0]),
                             TensorValue(input_layout.size.feature[0]),
                             TensorValue(input_layout.size.spatial[0]),
                             TensorValue(input_layout.size.spatial[1]),
                             TensorValue(input_layout.size.spatial[2]))};
    } else {
        return layout{output_type,
                      input_format,
                      tensor(TensorValue(input_layout.size.batch[0]),
                             TensorValue(input_layout.size.feature[0]),
                             TensorValue(input_layout.size.spatial[0]),
                             TensorValue(input_layout.size.spatial[1]))};
    }
}

std::string bucketize_inst::to_string(bucketize_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    std::string output_type = (desc->output_type == data_types::i64) ? "i64" : "i32";

    json_composite bucketize_info;
    bucketize_info.add("input id", input.id());
    bucketize_info.add("output_type", output_type);
    bucketize_info.add("with_right_bound", desc->with_right_bound);

    node_info->add("bucketize info", bucketize_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

bucketize_inst::typed_primitive_inst(network& network, bucketize_node const& node) : parent(network, node) {}

}  // namespace cldnn
