// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_copy_b.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"
#include "snippets/runtime_configurator.hpp"

namespace ov::intel_cpu {

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
    static VectorDims get_blk_order(size_t shape_rank);
    static VectorDims get_blk_shape(const VectorDims& planar_shape, ov::element::Type prc, bool is_transposed);

    void update_kernel(const RepackExecutorPtr& executor,
                       const VectorDims& shape,
                       const VectorDims& layout,
                       size_t N,
                       size_t K,
                       ov::element::Type prc);

    static const size_t brgemm_kernel_rank;
    std::unordered_map<size_t, RepackExecutorPtr> m_executors;
};

}  // namespace ov::intel_cpu
