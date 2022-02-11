// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <experimental_detectron_topk_rois_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>
#include <data_inst.h>

namespace cldnn {

primitive_type_id experimental_detectron_topk_rois::type_id() {
    static primitive_type_base<experimental_detectron_topk_rois> instance;
    return &instance;
}

experimental_detectron_topk_rois_inst::typed_primitive_inst(network& network, experimental_detectron_topk_rois_node const &node)
: parent(network, node) {
}

layout experimental_detectron_topk_rois_inst::calc_output_layout(experimental_detectron_topk_rois_node const &node) {
    auto primitive = node.get_primitive();
    auto input_layout = node.input(0).get_output_layout();

    int32_t roi_num = std::min(input_layout.size.sizes()[0], static_cast<int32_t>(node.get_primitive()->max_rois));

    return {input_layout.data_type, input_layout.format,  {roi_num,
                                                                 input_layout.size.sizes()[1], 1, 1 }};
}

std::string experimental_detectron_topk_rois_inst::to_string(experimental_detectron_topk_rois_node const &node) {
    auto node_info = node.desc_to_json();
    json_composite experimental_detectron_topk_rois_info;
    experimental_detectron_topk_rois_info.add("input id", node.input().id());
    experimental_detectron_topk_rois_info.add("indices id", node.input(1).id());
    experimental_detectron_topk_rois_info.add("max_rois", node.get_primitive()->max_rois);
    node_info->add("experimental detectron TopK ROIs info", experimental_detectron_topk_rois_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

} // namespace cldnn
