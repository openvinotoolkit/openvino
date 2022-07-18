// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorg_yolo_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id reorg_yolo::type_id() {
    static primitive_type_base<reorg_yolo> instance;
    return &instance;
}

layout reorg_yolo_inst::calc_output_layout(reorg_yolo_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for "
           "reorg_yolo_node!");
    auto input_layout = node.input().get_output_layout();
    auto desc = node.get_primitive();
    auto stride = desc->stride;

    cldnn::layout layoutTemp = cldnn::layout(input_layout.data_type,
                                             input_layout.format,
                                             tensor(input_layout.batch(),
                                                    input_layout.feature() * stride * stride,
                                                    input_layout.spatial(0) / stride,
                                                    input_layout.spatial(1) / stride));
    return layoutTemp;
}

std::string reorg_yolo_inst::to_string(reorg_yolo_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto stride = desc->stride;

    std::stringstream primitive_description;

    json_composite reorg_yolo_info;
    reorg_yolo_info.add("stride", stride);

    node_info->add("reorg yolo info", reorg_yolo_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
reorg_yolo_inst::typed_primitive_inst(network& network, reorg_yolo_node const& node) : parent(network, node) {}
}  // namespace cldnn
