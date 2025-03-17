// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/repacked_input.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/runtime_configurator.hpp"

namespace ov::intel_cpu {

class CPURuntimeConfig : public ov::snippets::RuntimeConfig {
public:
    OPENVINO_RTTI("CPURuntimeConfig", "0", ov::snippets::RuntimeConfig)
    CPURuntimeConfig() = default;

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

    enum class RepackingImplType {
        NONE,         // no kernel-outside repacking
        IN_PARALLEL,  // should be executed in parallel_nt by each thread
        SEPARATE,     // should be separathy from kernel executed
    };
    RepackingImplType repacking_impl_type = RepackingImplType::NONE;

    std::unordered_map<size_t, RepackedInput> repacked_inputs = {};
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

    // Note: This method is temporarily used only by `BrgemmExternalRepackingAdjuster` to create kernels for repacking.
    //       Please, remove this method when the adjuster is deprecated
    [[nodiscard]] const ov::intel_cpu::MultiCacheWeakPtr& get_cache() const {
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

}  // namespace ov::intel_cpu
