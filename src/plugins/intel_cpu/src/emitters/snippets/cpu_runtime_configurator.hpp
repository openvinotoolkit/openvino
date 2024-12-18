// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_copy_b.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/runtime_configurator.hpp"

namespace ov {
namespace intel_cpu {

class CPURuntimeConfig : public ov::snippets::RuntimeConfig {
public:
    OPENVINO_RTTI("CPURuntimeConfig", "0", ov::snippets::RuntimeConfig)
    CPURuntimeConfig() = default;

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

#if defined(OPENVINO_ARCH_X86_64)
    struct RepackedInput {
        RepackedInput() = default;
        RepackedInput(CpuBlockedMemoryDescPtr desc_,
                      std::shared_ptr<BrgemmCopyBKernelExecutor> executor_,
                      VectorDims in_offsets_,
                      VectorDims out_offsets_)
            : desc(std::move(desc_)),
              executor(std::move(executor_)),
              in_offsets(std::move(in_offsets_)),
              out_offsets(std::move(out_offsets_)) {}

        CpuBlockedMemoryDescPtr desc{nullptr};
        std::shared_ptr<BrgemmCopyBKernelExecutor> executor{nullptr};
        VectorDims in_offsets{};
        VectorDims out_offsets{};
    };

    std::unordered_map<size_t, RepackedInput> repacked_inputs = {};
#endif  // OPENVINO_ARCH_X86_64
    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};
};

class CPURuntimeConfigurator : public ov::snippets::RuntimeConfigurator {
public:
    CPURuntimeConfigurator(ov::intel_cpu::MultiCacheWeakPtr cache);

    /**
     * @brief Calculate Loop parameters of Loop emitters and update these values in CPURuntimeConfig
     * @param linear_ir LinearIR
     */
    void update_loop_args(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const;

    const ov::intel_cpu::MultiCacheWeakPtr& get_cache() const {
        return compiled_kernel_cache;
    }

protected:
    void update(const ov::snippets::lowered::LinearIRCPtr& linear_ir) override;
    void update_tensor_rank(const ov::snippets::VectorDims& master_shape) const override;
    void init_tensor_rank(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const override;
    void initialization(const ov::snippets::lowered::LinearIRCPtr& linear_ir) override;

    static const size_t rank6D;

    ov::intel_cpu::MultiCacheWeakPtr compiled_kernel_cache;
};

}  // namespace intel_cpu
}  // namespace ov
