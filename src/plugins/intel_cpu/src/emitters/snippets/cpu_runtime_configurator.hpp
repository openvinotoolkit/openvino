// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/runtime_configurator.hpp"

#include "snippets/lowered/port_descriptor.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"

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
};

class CPURuntimeConfigurator : public ov::snippets::RuntimeConfigurator {
public:
    CPURuntimeConfigurator();

protected:
    /**
     * @brief Update RuntimeConfig based on LinearIR
     * @param linear_ir LinearIR
     */
    void update(const ov::snippets::lowered::LinearIRCPtr& linear_ir) override;
    /**
     * @brief Update tensor rank based on master shape
     * @param master_shape Master shape
     */
    void update_tensor_rank(const ov::snippets::VectorDims& master_shape) override;
    /**
     * @brief Initializes tensor rank of config
     * @param linear_ir LinearIR
     */
    void init_tensor_rank(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const override;
    /**
     * @brief Calculate Loop parameters of Loop emitters and update these values in CPURuntimeConfig
     * @param linear_ir LinearIR
     */
    void update_loop_args(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const;

    static const size_t rank6D;
};

}   // namespace intel_cpu
}   // namespace ov
