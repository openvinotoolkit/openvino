// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime_optimizer.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

class MHAParallelWAOptimizer : public lowered::pass::RuntimeOptimizer {
public:
    MHAParallelWAOptimizer() = default;
    MHAParallelWAOptimizer(const lowered::LinearIRCPtr& linear_ir, RuntimeConfigurator* configurator);

    bool run(const lowered::LinearIR& linear_ir) override;

private:
    static std::unordered_set<lowered::ExpressionPtr> find_applicable_brgemms(const lowered::LinearIRCPtr& linear_ir);
    static std::unordered_set<size_t> find_unsqueezed_params(
        const lowered::LinearIRCPtr& linear_ir,
        const std::unordered_set<lowered::ExpressionPtr>& brgemms);
    static std::vector<lowered::ExpandedLoopInfoPtr> find_loops_to_split(
        const lowered::LinearIRCPtr& linear_ir,
        const std::unordered_set<size_t>& unsqueezed_params);

    std::vector<lowered::ExpandedLoopInfoPtr> loops_to_split{};
    std::unordered_set<size_t> unsqueezed_params{};
    std::vector<std::vector<size_t>> optimized_layouts{};
    std::vector<size_t> m_dim_idces{};
    size_t concurrency = 0;

    static const size_t m_dim_idx;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov