// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace snippets {

/**
 * @interface RuntimeConfig
 * @brief The config that contains information about LinearIR in runtime.
 */
class RuntimeConfig {
public:
    RuntimeConfig() = default;
    virtual ~RuntimeConfig() = default;

    // Note that get_type_info_static and get_type_info are needed to mimic OPENVINO_RTTI interface,
    // so the standard OPENVINO_RTTI(...) macros could be used in derived classes.
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static ::ov::DiscreteTypeInfo type_info_static {"RuntimeConfig"};
        type_info_static.hash();
        return type_info_static;
    }

    virtual const DiscreteTypeInfo& get_type_info() const {
        return get_type_info_static();
    }

    const char* get_type_name() const {
        return get_type_info().name;
    }
};

/**
 * @interface RuntimeConfigurator
 * @brief Configure runtime config based on runtime information of LinearIR
 */
class RuntimeConfigurator {
public:
    RuntimeConfigurator(std::shared_ptr<RuntimeConfig> c);
    virtual ~RuntimeConfigurator() = default;

    /**
     * @brief Update RuntimeConfig based on new state of LinearIR and return its
     * @param linear_ir LinearIR
     * @return updated config
     */
    const std::shared_ptr<RuntimeConfig>& get_updated_config(const std::shared_ptr<lowered::LinearIR>& linear_ir);

protected:
    /**
     * @brief Return `True` if config should be updated. Otherwise returns `False`
     * @param linear_ir LinearIR
     * @return boolean
     */
    virtual bool is_update_needed(const std::shared_ptr<lowered::LinearIR>& linear_ir) = 0;
    /**
     * @brief Update RuntimeConfig based on LinearIR
     * @param linear_ir LinearIR
     */
    virtual void update(const std::shared_ptr<lowered::LinearIR>& linear_ir) = 0;
    /**
     * @brief Update LinearIR parameters that depends on shape: LoopInfo in LoopManager
     * @param linear_ir LinearIR
     */
    void update_linear_ir_state(const std::shared_ptr<lowered::LinearIR>& linear_ir) const;

    std::shared_ptr<RuntimeConfig> m_config = nullptr;
    lowered::pass::PassPipeline m_state_updater;
};

} // namespace snippets
} // namespace ov
