// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/subgraph_manager.hpp"

#include <memory>

#include "snippets/op/subgraph.hpp"
#include "snippets/pass/common_optimizations.hpp"

namespace ov::snippets::pass {

bool CommonOptimizations::SubgraphManager::run_passes(const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph) {
    bool updated = false;
    for (const auto& pass : m_pass_list) {
        updated = pass->run_on_subgraph(subgraph) || updated;
    }
    return updated;
}

}  // namespace ov::snippets::pass
