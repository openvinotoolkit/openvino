// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vl_sdpa_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(vl_sdpa);

std::string vl_sdpa_inst::to_string(const vl_sdpa_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite vlsdpa_info;
    vlsdpa_info.add("q", node.input(0).id());
    vlsdpa_info.add("k", node.input(1).id());
    vlsdpa_info.add("v", node.input(2).id());
    vlsdpa_info.add("cu_seq_lens", node.input(3).id());

    node_info->add("vlsdpa_info", vlsdpa_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

vl_sdpa_inst::typed_primitive_inst(network& network, const vl_sdpa_node& node) : parent(network, node) {}

}  // namespace cldnn
