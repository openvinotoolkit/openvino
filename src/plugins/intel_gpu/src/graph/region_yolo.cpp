// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "region_yolo_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(region_yolo)

layout region_yolo_inst::calc_output_layout(region_yolo_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for "
           "region_yolo_node!");
    auto input_layout = impl_param.get_input_layout();
    auto desc = impl_param.typed_desc<region_yolo>();

    if (desc->do_softmax) {
        return cldnn::layout(
            input_layout.data_type,
            input_layout.format,
            tensor(input_layout.batch(),
                   input_layout.feature() * input_layout.spatial(0) * input_layout.spatial(1),
                   1,
                   1));
    } else {
        tensor::value_type features = (desc->classes + desc->coords + 1) * desc->mask_size;
        return cldnn::layout(
            input_layout.data_type,
            input_layout.format,
            tensor(input_layout.batch(), features, input_layout.spatial(0), input_layout.spatial(1)));
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
