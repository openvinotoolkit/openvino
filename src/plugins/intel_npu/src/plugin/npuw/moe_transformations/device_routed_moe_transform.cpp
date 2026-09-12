// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_routed_moe_transform.hpp"

#include "../logging.hpp"
#include "moe_topology.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov::npuw::pass {

bool DeviceRoutedMoETransform::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    const ov::npuw::moe::BatchedMoEPattern pattern;
    for (const auto& node : model->get_ordered_ops()) {
        const auto topology = pattern.match(node);
        if (!topology)
            continue;
        // Build first, replace last: an unsupported expression must never leave
        // a partially rewritten layer or mutate a shared weight/score producer.
        auto replacement = ov::npuw::moe::build_device_routed_moe(*topology);
        if (!replacement)
            continue;
        ov::replace_node(topology->reduction, replacement);
        LOG_INFO("DeviceRoutedMoE: selected " << topology->num_selected << "/" << topology->num_experts
                                               << " experts for " << replacement->get_friendly_name());
        changed = true;
    }
    return changed;
}

}  // namespace ov::npuw::pass
