// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id gather::type_id() {
    static primitive_type_base<gather> instance;
    return &instance;
}

layout gather_inst::calc_output_layout(gather_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    std::vector<tensor::value_type> dims_converted(desc->output_shape.begin(), desc->output_shape.end());
    // extend shape to 4d
    for (size_t i = dims_converted.size(); i < 4; i++) {
        dims_converted.push_back(1);
    }

    auto output_format = input_layout.format;
    auto output_type = input_layout.data_type;
    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    //명세에는 output_shape가 어떤 포맷으로 들어오는지 명시되있지 않음.
    //지금까지 bfyx로 입력해서 잘 되었으므로 bfyx로 입력된다고 가정함.
    //base = bfyx
    auto d = dims_converted;
    //handle other orders
    if (output_format == format::yxfb) {
        dims_converted[0] = d[2];
        dims_converted[1] = d[3];
        dims_converted[2] = d[1];
        dims_converted[3] = d[0];
    } else if (output_format == format::fyxb) {
        dims_converted[0] = d[1];
        dims_converted[1] = d[2];
        dims_converted[2] = d[3];
        dims_converted[3] = d[0];
    } else if (output_format == format::fs_b_yx_fsv32) {
        dims_converted[0] = d[1];
        dims_converted[1] = d[0];
        dims_converted[2] = d[2];
        dims_converted[3] = d[3];
    }
    return layout{output_type, output_format, tensor(output_format, dims_converted)};
}

std::string gather_inst::to_string(gather_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_info;
    gather_info.add("input id", input.id());
    gather_info.add("axis", desc->axis);
    gather_info.add("batch_dim", desc->batch_dim);
    gather_info.add("output shape", cldnn::to_string(desc->output_shape));

    node_info->add("gather info", gather_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_inst::typed_primitive_inst(network& network, gather_node const& node) : parent(network, node) {}

}  // namespace cldnn
