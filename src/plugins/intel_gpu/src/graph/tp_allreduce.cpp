// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP_GPU

#include "tp_allreduce_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

#include <string>

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(tp_allreduce)

std::string tp_allreduce_inst::to_string(const tp_allreduce_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite tp_info;
    tp_info.add("collective_id", desc->collective_id);
    tp_info.add("rank", desc->rank);

    node_info->add("tp_allreduce params", tp_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

tp_allreduce_inst::typed_primitive_inst(network& network, const tp_allreduce_node& node)
    : parent(network, node) {}

}  // namespace cldnn

#endif  // ENABLE_TP_GPU
