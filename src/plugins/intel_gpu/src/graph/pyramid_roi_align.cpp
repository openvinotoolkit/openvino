// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "pyramid_roi_align_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id pyramid_roi_align::type_id() {
    static primitive_type_base<pyramid_roi_align> instance;
    return &instance;
}

layout pyramid_roi_align_inst::calc_output_layout(pyramid_roi_align_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for "
           "pyramid_roi_align node!");

    auto desc = node.get_primitive();

    auto boxes_layout = node.input().get_output_layout();
    auto P2_layout = node.P2().get_output_layout();

    int32_t output_b = boxes_layout.batch();
    int32_t output_f = P2_layout.feature();

    int32_t output_x = desc->output_size;
    int32_t output_y = desc->output_size;

    return layout{P2_layout.data_type, P2_layout.format, {output_b, output_f, output_x, output_y}};
}

std::string pyramid_roi_align_inst::to_string(pyramid_roi_align_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    std::stringstream primitive_description;
    json_composite pyramid_roi_align_info;
    node_info->add("pyramid_roi_align_info", pyramid_roi_align_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

pyramid_roi_align_inst::typed_primitive_inst(network& network, pyramid_roi_align_node const& node)
    : parent(network, node) {}
}  // namespace cldnn
