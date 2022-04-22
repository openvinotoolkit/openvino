// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "region_yolo_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id region_yolo::type_id() {
    static primitive_type_base<region_yolo> instance;
    return &instance;
}

layout region_yolo_inst::calc_output_layout(region_yolo_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_types.at(0)) == false &&
           "Output data type forcing is not supported for "
           "region_yolo_node!");
    auto input_layout = node.input().get_output_layout();
    auto desc = node.get_primitive();

    if (desc->do_softmax) {
        return cldnn::layout(
            input_layout.data_type,
            input_layout.format,
            tensor(input_layout.size.batch[0],
                   input_layout.size.feature[0] * input_layout.size.spatial[0] * input_layout.size.spatial[1],
                   1,
                   1));
    } else {
        tensor::value_type features = (desc->classes + desc->coords + 1) * desc->mask_size;
        return cldnn::layout(
            input_layout.data_type,
            input_layout.format,
            tensor(input_layout.size.batch[0], features, input_layout.size.spatial[0], input_layout.size.spatial[1]));
    }
}

std::string region_yolo_inst::to_string(region_yolo_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto coords = desc->coords;
    auto classes = desc->classes;
    auto num = desc->num;
    auto do_softmax = desc->do_softmax;
    auto mask_size = desc->mask_size;

    std::stringstream primitive_description;

    json_composite region_yolo_info;
    region_yolo_info.add("coords", coords);
    region_yolo_info.add("classes", classes);
    region_yolo_info.add("num", num);
    region_yolo_info.add("do_softmax", do_softmax);
    region_yolo_info.add("mask_size", mask_size);

    node_info->add("region yolo info", region_yolo_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

region_yolo_inst::typed_primitive_inst(network& network, region_yolo_node const& node) : parent(network, node) {}
}  // namespace cldnn
