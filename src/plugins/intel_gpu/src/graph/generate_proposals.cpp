// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generate_proposals_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(generate_proposals)

layout generate_proposals_inst::calc_output_layout(const generate_proposals_node& node, kernel_impl_params const& impl_param) {
    const layout data_layout = impl_param.get_input_layout();
    const auto num_batches = data_layout.batch();
    const auto desc = impl_param.typed_desc<generate_proposals>();
    return layout(data_layout.data_type, data_layout.format, {static_cast<int>(num_batches * desc->post_nms_count), 4, 1, 1});
}

std::string generate_proposals_inst::to_string(const generate_proposals_node& node) {
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    info.add("min_size", desc->min_size);
    info.add("nms_threshold", desc->nms_threshold);
    info.add("pre_nms_count", desc->pre_nms_count);
    info.add("post_nms_count", desc->post_nms_count);
    info.add("normalized", desc->normalized);
    info.add("nms_eta", desc->nms_eta);

    auto node_info = node.desc_to_json();
    node_info->add("generate_proposals_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
}  // namespace cldnn
