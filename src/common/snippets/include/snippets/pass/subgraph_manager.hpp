// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <typeinfo>
#include <vector>

#include "snippets/pass/common_optimizations.hpp"

#include "snippets/pass/subgraph_pass.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace snippets {
namespace pass {
/**
 * @brief Manager class allows to manage transformation passes (SubgraphPasses) on Subgraph ops.
 *        See SubgraphPasses description for more details.
 *        It's light version of ov::Manager implementation the purpose of which is to change only Subgraph as separate node in model.
 * @ingroup snippets
 */
class CommonOptimizations::SubgraphManager {
public:
    SubgraphManager() = default;

    /// @brief Register given transformation class type to execution list
    /// @return shared_ptr to the transformation instance
    template <typename T, class... Args>
    std::shared_ptr<T> register_pass(Args&&... args) {
        static_assert(std::is_base_of<SubgraphPass, T>::value, "pass not derived from SubgraphPass base");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        m_pass_list.push_back(std::static_pointer_cast<SubgraphPass>(pass));
        return pass;
    }

    /// @brief      Runs registered transformations on a given model
    /// @param      subgraph Input model
    /// @return     Returns true if the model was changed by transformations, false otherwise.
    bool run_passes(std::shared_ptr<ov::snippets::op::Subgraph> subgraph);

protected:
    std::vector<std::shared_ptr<SubgraphPass>> m_pass_list;
};
}  // namespace pass
}  // namespace snippets
}  // namespace ov
