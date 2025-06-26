// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "cache/multi_cache.h"
#include "cpu_types.h"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_copy_b.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"

namespace ov::intel_cpu::pass {

/**
 * @class BrgemmExternalRepackingAdjuster
 * @brief A runtime optimizer that creates the memory descs for BRGEMM inputs which require external repacking.
 * The generated memory descs are stored in the CPU runtime config.
 */
class BrgemmExternalRepackingAdjuster : public ov::snippets::lowered::pass::RuntimeOptimizer {
public:
    OPENVINO_RTTI("BrgemmExternalRepackingAdjuster", "", RuntimeOptimizer)
    BrgemmExternalRepackingAdjuster() = default;
    BrgemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                    const CPURuntimeConfigurator* configurator);

    bool run(const snippets::lowered::LinearIR& linear_ir) override;
    bool applicable() const override {
        return !m_executors.empty();
    }

private:
    using RepackExecutorPtr = std::shared_ptr<BrgemmCopyBKernelExecutor>;
    static CpuBlockedMemoryDescPtr get_desc(const ov::snippets::VectorDims& planar_shape,
                                            const ov::element::Type& prc,
                                            size_t wei_k_blk,
                                            size_t wei_n_blk,
                                            bool are_wei_blocked,
                                            bool is_transposed);

    static void update_kernel(const RepackExecutorPtr& executor,
                              const VectorDims& shape,
                              const VectorDims& layout,
                              size_t N,
                              size_t K);

    static RepackExecutorPtr create_executor(const ov::snippets::lowered::ExpressionPtr& param,
                                             const ov::intel_cpu::MultiCacheWeakPtr& cache);

    static const size_t brgemm_kernel_rank;
    std::unordered_map<size_t, RepackExecutorPtr> m_executors;
};

}  // namespace ov::intel_cpu::pass
