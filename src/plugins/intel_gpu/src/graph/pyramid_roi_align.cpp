// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "pyramid_roi_align_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(pyramid_roi_align)

layout pyramid_roi_align_inst::calc_output_layout(pyramid_roi_align_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for "
           "pyramid_roi_align node!");

    auto desc = impl_param.typed_desc<pyramid_roi_align>();

    auto boxes_layout = impl_param.get_input_layout(0);
    auto P2_layout = impl_param.get_input_layout(1);

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
    node_info->add("pyramid_roi_align_info", std::move(pyramid_roi_align_info));
    node_info->dump(primitive_description);
    return primitive_description.str();
}

pyramid_roi_align_inst::typed_primitive_inst(network& network, pyramid_roi_align_node const& node)
    : parent(network, node) {}
}  // namespace cldnn
