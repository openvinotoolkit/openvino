// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/subgraph_manager.hpp"

namespace ov {
namespace snippets {
namespace pass {

bool CommonOptimizations::SubgraphManager::run_passes(std::shared_ptr<ov::snippets::op::Subgraph> subgraph) {
    bool updated = false;
    for (const auto& pass : m_pass_list) {
        updated = pass->run_on_subgraph(subgraph) || updated;
    }
    return updated;
}

} // namespace pass
} // namespace snippets
} // namespace ov
