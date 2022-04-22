// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <roi_align_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

namespace cldnn {

primitive_type_id roi_align::type_id() {
    static primitive_type_base<roi_align> instance;
    return &instance;
}

roi_align_inst::typed_primitive_inst(network& network, roi_align_node const& node)
    : parent(network, node) {}

layout roi_align_inst::calc_output_layout(roi_align_node const& node) {
    auto primitive = node.get_primitive();
    auto input_layout = node.input(0).get_output_layout();
    auto rois_layout = node.input(1).get_output_layout();
    auto num_rois = rois_layout.size.batch[0];
    auto num_channels = input_layout.size.feature[0];
    return layout(input_layout.data_type, format::bfyx, {num_rois, num_channels, primitive->pooled_h, primitive->pooled_w});
}

std::string roi_align_inst::to_string(roi_align_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite roi_align_info;
    roi_align_info.add("input id", node.input().id());
    roi_align_info.add("rois id", node.get_dependency(1).first->id());
    roi_align_info.add("batches id", node.get_dependency(2).first->id());
    roi_align_info.add("pooled_h", node.get_primitive()->pooled_h);
    roi_align_info.add("pooled_w", node.get_primitive()->pooled_w);
    roi_align_info.add("sampling_ratio", node.get_primitive()->sampling_ratio);
    roi_align_info.add("spatial_scale", node.get_primitive()->spatial_scale);
    roi_align_info.add("mode", node.get_primitive()->mode == roi_align::PoolingMode::Max ? "Max" : "Avg");
    node_info->add("roi_align info", roi_align_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

} // namespace cldnn
