// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/snippets/jit_snippets_call_args.hpp"
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

    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};
    std::unordered_map<size_t, CpuBlockedMemoryDescPtr> m_in_requested_descs = {};
};

class CPURuntimeConfigurator : public ov::snippets::RuntimeConfigurator {
public:
    CPURuntimeConfigurator();

    /**
     * @brief Calculate Loop parameters of Loop emitters and update these values in CPURuntimeConfig
     * @param linear_ir LinearIR
     */
    void update_loop_args(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const;
protected:
    void update(const ov::snippets::lowered::LinearIRCPtr& linear_ir) override;
    void update_tensor_rank(const ov::snippets::VectorDims& master_shape) const override;
    void init_tensor_rank(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const override;
    void initialization(const ov::snippets::lowered::LinearIRCPtr& linear_ir) override;

    static const size_t rank6D;
};

}   // namespace intel_cpu
}   // namespace ov
