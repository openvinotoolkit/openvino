// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include "emitters/snippets/aarch64/kernel_executors/gemm_copy_b.hpp"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"

namespace ov::intel_cpu::pass::aarch64 {

/**
 * @class GemmExternalRepackingAdjuster
 * @brief A runtime optimizer that configures external repacking for aarch64 GEMM inputs.
 */
class GemmExternalRepackingAdjuster : public ov::snippets::lowered::pass::RuntimeOptimizer {
public:
    OPENVINO_RTTI("GemmExternalRepackingAdjuster", "", RuntimeOptimizer)
    GemmExternalRepackingAdjuster() = default;
    GemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                  const CPURuntimeConfigurator* configurator);

    bool run(const snippets::lowered::LinearIR& linear_ir) override;
    bool applicable() const override {
        return !m_repacked_inputs.empty();
    }

private:
    using RepackExecutorPtr = std::shared_ptr<ov::intel_cpu::aarch64::GemmCopyBKernel>;
    struct RepackedInputConfig {
        bool needs_runtime_repacking = false;
    };

    static CpuBlockedMemoryDescPtr get_desc(const ov::snippets::VectorDims& planar_shape, const ov::element::Type& prc);

    static void update_kernel(const RepackExecutorPtr& executor,
                              const ov::snippets::VectorDims& shape,
                              const ov::snippets::VectorDims& layout,
                              size_t N,
                              size_t K,
                              const ov::element::Type& prc);

    static RepackExecutorPtr create_executor(const ov::element::Type& prc);

    static const size_t gemm_kernel_rank;
    std::unordered_map<size_t, RepackedInputConfig> m_repacked_inputs;
};

}  // namespace ov::intel_cpu::pass::aarch64
