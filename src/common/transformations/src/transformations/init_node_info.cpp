// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/init_node_info.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "transformations/utils/utils.hpp"

bool ov::pass::InitNodeInfo::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(InitNodeInfo);

    for (auto& node : f->get_ops()) {
        // Recursively apply transformation for sub-graph based operations
        ov::op::util::process_subgraph(*this, node);

        auto& rtInfo = node->get_rt_info();
        rtInfo.emplace(FusedNames::get_type_info_static(), FusedNames{node->get_friendly_name()});
    }
    return false;
}
