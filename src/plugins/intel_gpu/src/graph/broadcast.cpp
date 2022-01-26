// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_inst.h"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <vector>
#include <set>

namespace cldnn {
primitive_type_id broadcast::type_id() {
    static primitive_type_base<broadcast> instance;
    return &instance;
}

layout broadcast_inst::calc_output_layout(broadcast_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for broadcast_node!");
    auto input_layout = node.input().get_output_layout();
    auto desc = node.get_primitive();

    return {input_layout.data_type, input_layout.format, desc->broadcast_sizes};
}

std::string broadcast_inst::to_string(broadcast_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    const auto& broadcast_sizes = desc->broadcast_sizes;
    const auto& broadcast_axes = desc->broadcast_axes;
    auto& input = node.input();

    std::stringstream primitive_description;
    std::stringstream ss_broadcast_axes;

    for (size_t i = 0; i < broadcast_axes.size(); ++i) {
        ss_broadcast_axes << broadcast_axes.at(i);
        i != (broadcast_axes.size() - 1) ? ss_broadcast_axes << ", " : ss_broadcast_axes << "";
    }

    json_composite broadcast_info;
    broadcast_info.add("input id", input.id());
    broadcast_info.add("broadcast_sizes", broadcast_sizes.to_string());
    broadcast_info.add("broadcast axes", ss_broadcast_axes.str());

    node_info->add("broadcast info", broadcast_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

broadcast_inst::typed_primitive_inst(network& network, broadcast_node const& node) : parent(network, node) {
    auto input_layout = node.input().get_output_layout();

    const auto& input_sizes = input_layout.size;
    const auto& output_sizes = argument.broadcast_sizes;
    const auto format = input_layout.format;

    std::vector<tensor::value_type> input_dims;
    size_t max_axes_num;

    if (format == format::bfzyx) {
        max_axes_num = 5;
        input_dims = {input_sizes.batch[0],
                      input_sizes.feature[0],
                      input_sizes.spatial[2],
                      input_sizes.spatial[1],
                      input_sizes.spatial[0]};
    } else {
        max_axes_num = 4;
        input_dims = {input_sizes.batch[0], input_sizes.feature[0], input_sizes.spatial[1], input_sizes.spatial[0]};
    }

    std::vector<tensor::value_type> reordered_input_dims(5, 0);
    std::set<uint16_t> existing;

    const auto& broadcast_axes = node.get_primitive()->broadcast_axes;
    size_t broadcast_axes_size = broadcast_axes.size();
    size_t index = 0;
    size_t input_index = broadcast_axes_size;

    if (format == format::bfzyx) {
        if (broadcast_axes_size > 5) {
            CLDNN_ERROR_MESSAGE(node.id(),
                                "Incorrect parameters configuration: broadcast_axes size should be less or equal 5.");
        }
    } else if (broadcast_axes_size > 4) {
        CLDNN_ERROR_MESSAGE(node.id(),
                            "Incorrect parameters configuration: broadcast_axes size should be less or equal 4.");
    }
    for (size_t i = 0; i < broadcast_axes_size; ++i) {
        if (broadcast_axes.at(i) >= max_axes_num) {
            CLDNN_ERROR_MESSAGE(
                node.id(),
                "Incorrect parameters configuration: broadcast_axes index should be within broadcast_sizes range.");
        }
        if (existing.find(broadcast_axes.at(i)) != existing.end()) {
            CLDNN_ERROR_MESSAGE(
                node.id(),
                "Incorrect parameters configuration: Duplicate axes numbers was found in broadcast_axes.");
        }
        existing.insert(broadcast_axes.at(i));
    }
    for (size_t i = 0; i < input_index; ++i) {
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Input size on dimension number " + std::to_string(i),
                              input_dims.at(i),
                              "",
                              1,
                              "Must be equal 1.");
    }
    // bfyx, bfzyx format
    for (size_t i = 0; i < max_axes_num; ++i) {
        if (std::find(broadcast_axes.begin(), broadcast_axes.end(), i) != broadcast_axes.end()) {
            reordered_input_dims.at(i) = input_dims.at(index);
            ++index;
        } else {
            reordered_input_dims.at(i) = input_dims.at(input_index);
            ++input_index;
        }
    }
    tensor input_sizes_to_compare;
    if (format == format::bfzyx)
        input_sizes_to_compare = {reordered_input_dims.at(0),
                                  reordered_input_dims.at(1),
                                  reordered_input_dims.at(4),
                                  reordered_input_dims.at(3),
                                  reordered_input_dims.at(2)};
    else
        input_sizes_to_compare = {reordered_input_dims.at(0),
                                  reordered_input_dims.at(1),
                                  reordered_input_dims.at(3),
                                  reordered_input_dims.at(2)};

    CLDNN_ERROR_TENSOR_SIZES_NOT_DIVIDABLE(node.id(),
                                           "Broadcast sizes",
                                           output_sizes,
                                           "input sizes",
                                           input_sizes_to_compare,
                                           "Invalid broadcast size: not dividable by input size");
}
}  // namespace cldnn
