// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_set>

#include "openvino/core/node.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::gguf {
namespace pass {

// Records which model Parameters currently have a live consumer, for a paired
// PruneParametersOrphanedSince to compare against later.
class SnapshotLiveParameters : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::SnapshotLiveParameters");
    explicit SnapshotLiveParameters(std::shared_ptr<std::unordered_set<const ov::Node*>> out);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<std::unordered_set<const ov::Node*>> m_out;
};

// Drops a Parameter that was live in the paired snapshot but no longer is, without touching one
// some earlier, unrelated pass already left disconnected.
class PruneParametersOrphanedSince : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::PruneParametersOrphanedSince");
    explicit PruneParametersOrphanedSince(std::shared_ptr<std::unordered_set<const ov::Node*>> live_before);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<std::unordered_set<const ov::Node*>> m_live_before;
};

}  // namespace pass
}  // namespace ov::frontend::gguf
