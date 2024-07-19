// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"

#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>

using namespace cldnn;

namespace cldnn {

void graph_initializations::set_outputs(program& p) {
    auto custom_outputs = p.get_config().get_property(ov::intel_gpu::custom_outputs);
    if (!custom_outputs.empty()) {
        for (auto const& output : custom_outputs) {
            OPENVINO_ASSERT(p.has_node(output), "not found custom output node in current cldnn::program: ", output);
            auto o_node = p.get_node_ptr(output);
            o_node->set_output(true);
            p.outputs.push_back(o_node.get());
        }
    } else {
        for (auto& node : p.nodes_map)
            if (node.second->is_endpoint() && !node.second->is_type<data>()) {
                node.second->set_output(true);
                p.outputs.push_back(node.second.get());
            }
    }
}

void graph_initializations::run(program& p) {
    set_outputs(p);

    auto forcing_map = p.get_config().get_property(ov::intel_gpu::force_implementations);
    for (auto& kv : forcing_map) {
        if (p.has_node(kv.first)) {
            p.get_node(kv.first).set_forced_impl_type(kv.second.impl_type);
        }
    }

    p.get_processing_order().calc_processing_order(p);
}
}  // namespace cldnn
