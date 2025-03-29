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
 * @class MHAParallelWAOptimizer
 * @brief Optimizes the dynamic MHA execution increasing parallel work amount dy dividing Brgemm's "M" dimension to "parallel_m"
 * and "kernel_m". Uses heuristics from snippets::pass::SplitDimensionM for dimension splitting.
 * The optimizer performs the following steps:
 * - Identifies applicable Brgemm operations within the LinearIR.
 * - Finds parameters whose shapes and layouts need to be adjusted after the split.
 * - Determines loops that should be adjusted.
 */
class MHAParallelWAOptimizer : public lowered::pass::RuntimeOptimizer {
    friend class SetDynamicWAToOuterMostLoop;
public:
    OPENVINO_RTTI("MHAParallelWAOptimizer", "", RuntimeOptimizer)
    MHAParallelWAOptimizer() = default;
    MHAParallelWAOptimizer(const lowered::LinearIRCPtr& linear_ir, const RuntimeConfigurator* configurator);

    bool run(const lowered::LinearIR& linear_ir) override;
    bool applicable() const override { return !m_loops_to_split.empty(); }

private:
    static std::unordered_set<lowered::ExpressionPtr> find_applicable_brgemms(
        const lowered::LinearIRCPtr& linear_ir,
        bool check_dynamic_wa = true);

    static std::unordered_set<size_t> find_unsqueezed_params(
        const lowered::LinearIRCPtr& linear_ir,
        const std::unordered_set<lowered::ExpressionPtr>& brgemms);

    static std::vector<lowered::ExpandedLoopInfoPtr> find_loops_to_split(
        const lowered::LinearIRCPtr& linear_ir,
        const std::unordered_set<size_t>& unsqueezed_params);

    std::vector<lowered::ExpandedLoopInfoPtr> m_loops_to_split{};
    std::unordered_set<size_t> m_unsqueezed_params{};
    std::vector<std::vector<size_t>> m_optimized_layouts{};
    std::vector<size_t> m_dim_M_idces{};
    size_t m_concurrency = 0;

    static const size_t m_dim_M_idx;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov