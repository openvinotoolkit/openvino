// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/init_node_info.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

bool ov::pass::InitNodeInfo::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(InitNodeInfo);

    for (auto& node : f->get_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                run_on_model(sub_graph);
            }
        }
        auto& rtInfo = node->get_rt_info();
        rtInfo.emplace(FusedNames::get_type_info_static(), FusedNames{node->get_friendly_name()});
        if (ov::fp16_compression_is_disabled(node))
            rtInfo.emplace(ov::DisableFP16Compression::get_type_info_static(), DisableFP16Compression{});
    }
    return false;
}
