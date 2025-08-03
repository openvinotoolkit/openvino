// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <unordered_set>
#include <vector>

#include "openvino/core/rtti.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"
#include "snippets/runtime_configurator.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @class FAParallelWAOptimizer
 * @brief Optimizes the dynamic FA execution increasing parallel work amount dy dividing Brgemm's "M" dimension to
 * "parallel_m" and "kernel_m". Uses heuristics from snippets::pass::SplitDimensionM for dimension splitting.
 * set new_m to rt info.
 */
class FAParallelWAOptimizer : public lowered::pass::RuntimeOptimizer {

public:
    OPENVINO_RTTI("FAParallelWAOptimizer", "", RuntimeOptimizer)
    FAParallelWAOptimizer() = default;
    FAParallelWAOptimizer(const lowered::LinearIRCPtr& linear_ir, const RuntimeConfigurator* configurator);

    bool run(const lowered::LinearIR& linear_ir) override;
    bool applicable() const override {
        return has_dynamic_fa;
    }

private:
    static std::unordered_set<size_t> find_unsqueezed_params(const lowered::LinearIRCPtr& linear_ir,
                                                             const std::unordered_set<lowered::ExpressionPtr>& fas);
    bool has_dynamic_fa = false;
    std::unordered_set<size_t> m_unsqueezed_params;
    std::vector<std::vector<size_t>> m_optimized_layouts;
    std::vector<size_t> m_dim_M_idces;
    size_t m_concurrency = 0;

    static const size_t m_dim_M_idx;
};

}  // namespace ov::snippets::lowered::pass
