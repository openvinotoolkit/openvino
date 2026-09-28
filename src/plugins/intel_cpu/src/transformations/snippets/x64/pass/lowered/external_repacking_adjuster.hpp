// Copyright (C) 2018-2026 Intel Corporation
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
#include "transformations/snippets/common/pass/lowered/external_repacking_adjuster.hpp"

namespace ov::intel_cpu::pass {

/**
 * @class BrgemmExternalRepackingAdjuster
 * @brief A runtime optimizer that creates the memory descs for BRGEMM inputs which require external repacking.
 * The generated memory descs are stored in the CPU runtime config.
 */
class BrgemmExternalRepackingAdjuster : public ov::intel_cpu::pass::ExternalRepackingAdjusterBase {
public:
    OPENVINO_RTTI("BrgemmExternalRepackingAdjuster", "", ExternalRepackingAdjusterBase)
    BrgemmExternalRepackingAdjuster() = default;
    BrgemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                    const CPURuntimeConfigurator* configurator);

private:
    using RepackExecutorPtr = std::shared_ptr<BrgemmCopyBKernelExecutor>;
    struct RepackedInputConfig {
        BrgemmCopyBKernelConfig kernel_config;
        RepackExecutorPtr executor = nullptr;
    };

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

    static BrgemmCopyBKernelConfig get_kernel_config(const ov::snippets::lowered::ExpressionPtr& param);
    static RepackExecutorPtr create_executor(const BrgemmCopyBKernelConfig& kernel_config,
                                             const ov::intel_cpu::MultiCacheWeakPtr& cache);

    static const size_t brgemm_kernel_rank;
    std::unordered_map<size_t, RepackedInputConfig> m_repacked_input_configs;
};

}  // namespace ov::intel_cpu::pass
