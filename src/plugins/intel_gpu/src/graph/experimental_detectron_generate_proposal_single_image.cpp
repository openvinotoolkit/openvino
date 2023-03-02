// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_generate_proposals_single_image_inst.hpp"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(experimental_detectron_generate_proposals_single_image)

layout experimental_detectron_generate_proposals_single_image_inst::calc_output_layout(
        const experimental_detectron_generate_proposals_single_image_node& node, kernel_impl_params const& impl_param) {
    const layout data_layout = impl_param.get_input_layout();
    auto desc = impl_param.typed_desc<experimental_detectron_generate_proposals_single_image>();

    return layout(data_layout.data_type, data_layout.format, {static_cast<int>(desc->post_nms_count), 4, 1, 1});
}

std::string experimental_detectron_generate_proposals_single_image_inst::to_string(
        const experimental_detectron_generate_proposals_single_image_node& node) {
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite ed_info;
    ed_info.add("min_size", desc->min_size);
    ed_info.add("nms_threshold", desc->nms_threshold);
    ed_info.add("pre_nms_count", desc->pre_nms_count);
    ed_info.add("post_nms_count", desc->post_nms_count);

    auto node_info = node.desc_to_json();
    node_info->add("experimental_detectron_generate_proposals_single_image_info", ed_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
}  // namespace cldnn
