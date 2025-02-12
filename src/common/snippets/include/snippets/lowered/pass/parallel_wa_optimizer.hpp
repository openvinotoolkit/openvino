// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

class SetDynamicWAToOuterMostLoop;
/**
 * @class ParallelWAOptimizer
 * @brief Optimizes the dynamic MHA execution increasing parallel work amount dy dividing Brgemm's "M" dimension to "parallel_m"
 * and "kernel_m". Uses heuristics from snippets::pass::SplitDimensionM for dimension splitting.
 * The optimizer performs the following steps:
 * - Identifies applicable Brgemm operations within the LinearIR.
 * - Finds parameters whose shapes and layouts need to be adjusted after the split.
 * - Determines loops that should be adjusted.
 */
class ParallelWAOptimizer : public lowered::pass::RuntimeOptimizer {
    friend class SetDynamicWAToOuterMostLoop;
public:
    OPENVINO_RTTI("ParallelWAOptimizer", "", RuntimeOptimizer)
    ParallelWAOptimizer() = default;
    ParallelWAOptimizer(const lowered::LinearIRCPtr& linear_ir, const RuntimeConfigurator* configurator);

    bool run(const lowered::LinearIR& linear_ir) override;
    bool applicable() const override {
        if (std::getenv("REF"))
            return false;
        return !m_loops_to_split.empty();
    }

private:
    std::vector<lowered::ExpandedLoopInfoPtr> m_loops_to_split{};
    size_t m_concurrency = 0;
    std::vector<std::vector<size_t>> m_optimized_layouts{};
    std::unordered_set<size_t> m_unsqueezed_params{};
    std::vector<size_t> m_dim_M_idces{};
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov