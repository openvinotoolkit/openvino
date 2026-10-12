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
#include "transformations/snippets/common/pass/lowered/external_repacking_adjuster.hpp"

namespace ov::intel_cpu::pass::aarch64 {

/**
 * @class GemmExternalRepackingAdjuster
 * @brief A runtime optimizer that configures external repacking for aarch64 GEMM inputs.
 */
class GemmExternalRepackingAdjuster : public ov::intel_cpu::pass::ExternalRepackingAdjusterBase {
public:
    OPENVINO_RTTI("GemmExternalRepackingAdjuster", "", ExternalRepackingAdjusterBase)
    GemmExternalRepackingAdjuster() = default;
    GemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                  const CPURuntimeConfigurator* configurator);

private:
    using RepackExecutorPtr = std::shared_ptr<ov::intel_cpu::aarch64::GemmCopyBKernel>;

    size_t update_runtime_repacking_data_size(const snippets::lowered::LinearIR& linear_ir,
                                              const CPURuntimeConfig& cpu_config,
                                              size_t idx) override;
    void update_runtime_repacking_input(const snippets::lowered::LinearIR& linear_ir,
                                        CPURuntimeConfig& cpu_config,
                                        size_t idx,
                                        bool is_impl_parallel) override;
    void update_compile_time_repacked_input(const snippets::lowered::LinearIR& linear_ir,
                                            CPURuntimeConfig& cpu_config,
                                            size_t idx) override;

    static CpuBlockedMemoryDescPtr get_desc(const ov::snippets::VectorDims& planar_shape, const ov::element::Type& prc);

    static void update_kernel(const RepackExecutorPtr& executor,
                              const ov::snippets::VectorDims& shape,
                              const ov::snippets::VectorDims& layout,
                              size_t N,
                              size_t K,
                              const ov::element::Type& prc);

    static RepackExecutorPtr create_executor(const ov::element::Type& prc);

    static const size_t gemm_kernel_rank;
};

}  // namespace ov::intel_cpu::pass::aarch64
