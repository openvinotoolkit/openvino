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

    /**
     * @brief Tries to split M dimension in "shape" in accordance to optimal parallel work amount
     * @param shape Original shape
     * @param optimal_parallelism_work_amount Optimal work amount
     * @param batch_m_dim reference on batch's part of the split M
     * @param new_m_dim reference on new M dim after the split
     * @return true if split was successfull, otherwise false
     */
    static bool split(const ov::Shape& shape, size_t optimal_parallelism_work_amount, size_t& batch_m_dim, size_t& new_m_dim);

private:
    /**
     * @brief Contains splitM approaches allowing to get the batch ideally divisible by optimal_parallelism_work_amount
     */
    static std::pair<size_t, size_t> split_ideally(size_t batch_dim, size_t m_dim, size_t optimal_parallelism_work_amount);
    /**
     * @brief Splits m_dim to minimize kernel_m in order to reduce waiting time for idle threads at the last parallel loop iteration.
     */
    static std::pair<size_t, size_t> split_minimize_kernel_wa(size_t batch_dim, size_t m_dim, size_t optimal_parallelism_work_amount);
    /**
     * @brief Splits m_dim to get the batch in (optimal_parallelism_work_amount, 2 * optimal_parallelism_work_amount) interval
     */
    static std::pair<size_t, size_t> split_fallback_increase_parallel_wa(size_t batch_dim, size_t m_dim, size_t optimal_parallelism_work_amount);

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
    static const size_t m_min_kernel_m;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov