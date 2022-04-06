// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot_inst.h"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <vector>

namespace cldnn {
primitive_type_id one_hot::type_id() {
    static primitive_type_base<one_hot> instance;
    return &instance;
}

static bool is_output_bfzyx(const layout& input, int32_t axis) {
    if (input.format == format::bfzyx)
        return true;
    if (axis == 4)
        return true;
    auto in_dims = input.size.sizes(format::bfyx);
    if (in_dims[3] != 1)
        return true;
    return false;
}

layout one_hot_inst::calc_output_layout(one_hot_node const& node) {
    auto input_layout = node.input().get_output_layout();
    auto desc = node.get_primitive();

    auto dt = !desc->output_data_types.empty() ? *desc->output_data_types.at(0) : input_layout.data_type;
    auto format = input_layout.format;

    if (desc->one_hot_axis > 4) {
        CLDNN_ERROR_MESSAGE(node.id(),
                            "Incorrect parameters configuration: one_hot_axis should be less or equal to 4.");
    }

    if (is_output_bfzyx(input_layout, desc->one_hot_axis))
        format = format::bfzyx;

    return {dt, format, desc->shape};
}

std::string one_hot_inst::to_string(one_hot_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    const auto& shape = desc->shape;
    const auto& one_hot_axis = desc->one_hot_axis;
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite one_hot_info;
    one_hot_info.add("input id", input.id());
    one_hot_info.add("output shape", shape.to_string());
    one_hot_info.add("one-hot axis", one_hot_axis);

    node_info->add("one_hot info", one_hot_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

one_hot_inst::typed_primitive_inst(network& network, one_hot_node const& node) : parent(network, node) {
    auto input_layout = node.input().get_output_layout();

    const auto& input_sizes = input_layout.size;
    const auto& output_sizes = argument.shape;

    std::vector<tensor::value_type> input_dims = {input_sizes.batch[0],
                                                  input_sizes.feature[0],
                                                  input_sizes.spatial[1],
                                                  input_sizes.spatial[0]};
    std::vector<tensor::value_type> output_dims = {output_sizes.batch[0],
                                                   output_sizes.feature[0],
                                                   output_sizes.spatial[1],
                                                   output_sizes.spatial[0]};

    if (is_output_bfzyx(input_layout, node.get_primitive()->one_hot_axis)) {
        output_dims.insert(output_dims.begin() + 2, output_sizes.spatial[2]);
    }

    const auto& one_hot_axis = node.get_primitive()->one_hot_axis;

    for (size_t i = 0, j = 0; j < output_dims.size() - 1; ++i, ++j) {
        if (j == one_hot_axis)
            ++j;
        if (input_dims[i] != output_dims[j]) {
            CLDNN_ERROR_MESSAGE(node.id(), "Incorrect parameters configuration: shape does not fit input size.");
        }
    }
}
}  // namespace cldnn
