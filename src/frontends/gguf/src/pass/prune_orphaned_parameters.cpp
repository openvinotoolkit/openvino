// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/prune_orphaned_parameters.hpp"

#include <algorithm>

#include "openvino/core/model.hpp"

namespace ov::frontend::gguf {
namespace pass {
namespace {

// A Parameter can have non-empty target_inputs() yet be practically dead if its consumer isn't
// reachable from any Result/Sink. get_ordered_ops() is rooted at Results/Sinks, so use it instead
// of target_inputs() to find genuinely-live parameters.
std::unordered_set<const ov::Node*> live_parameters(const std::shared_ptr<ov::Model>& model) {
    std::unordered_set<const ov::Node*> ordered_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        ordered_nodes.insert(node.get());
    }
    std::unordered_set<const ov::Node*> live;
    for (const auto& param : model->get_parameters()) {
        const auto& targets = param->output(0).get_target_inputs();
        const bool has_live_consumer = std::any_of(targets.begin(), targets.end(), [&](const ov::Input<ov::Node>& in) {
            return ordered_nodes.count(in.get_node()) != 0;
        });
        if (has_live_consumer) {
            live.insert(param.get());
        }
    }
    return live;
}

}  // namespace

SnapshotLiveParameters::SnapshotLiveParameters(std::shared_ptr<std::unordered_set<const ov::Node*>> out)
    : m_out(std::move(out)) {}

bool SnapshotLiveParameters::run_on_model(const std::shared_ptr<ov::Model>& model) {
    *m_out = live_parameters(model);
    return false;
}

PruneParametersOrphanedSince::PruneParametersOrphanedSince(
    std::shared_ptr<std::unordered_set<const ov::Node*>> live_before)
    : m_live_before(std::move(live_before)) {}

bool PruneParametersOrphanedSince::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const auto live_after = live_parameters(model);
    // remove_parameter() mutates the model's parameter list, so snapshot it before iterating.
    const ov::ParameterVector parameters = model->get_parameters();
    bool changed = false;
    for (const auto& param : parameters) {
        if (m_live_before->count(param.get()) && !live_after.count(param.get())) {
            model->remove_parameter(param);
            changed = true;
        }
    }
    return changed;
}

}  // namespace pass
}  // namespace ov::frontend::gguf
