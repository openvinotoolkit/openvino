// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/any.hpp"
#include "openvino/runtime/plugin_config.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include <thread>

namespace ov::intel_gpu {

struct ExecutionConfig : public ov::PluginConfig {
    ExecutionConfig();
    ExecutionConfig(std::initializer_list<ov::AnyMap::value_type> values) : ExecutionConfig() { set_property(ov::AnyMap(values)); }
    explicit ExecutionConfig(const ov::AnyMap& properties) : ExecutionConfig() { set_property(properties); }
    explicit ExecutionConfig(const ov::AnyMap::value_type& property) : ExecutionConfig() { set_property(property); }

    // Default operators copy config as is including finalized flag state
    // In case if the config need updates after finalization clone() method shall be used as it resets finalized flag value.
    // That's needed to avoid unexpected options update as we call finalization twice: in transformation pipeline
    // and in cldnn::program c-tor (which is needed to handle unit tests mainly). So this second call may cause unwanted side effects
    // if config is not marked as finalized, which could have easily happened if copy operator reset finalization flag
    ExecutionConfig(const ExecutionConfig& other);
    ExecutionConfig& operator=(const ExecutionConfig& other);
    ExecutionConfig clone() const;

    void finalize(cldnn::engine& engine);
    using ov::PluginConfig::finalize;

    const ov::AnyMap& get_user_properties() const { return m_user_properties; }

protected:
    void finalize_impl(const IRemoteContext* context) override;
    void apply_model_specific_options(const IRemoteContext* context, const ov::Model& model) override;
    void apply_rt_info(const IRemoteContext* context, const ov::RTMap& rt_info, bool is_llm);

    void apply_user_properties(const cldnn::device_info& info);
    void apply_hints(const cldnn::device_info& info);
    void apply_execution_hints(const cldnn::device_info& info);
    void apply_performance_hints(const cldnn::device_info& info);
    void apply_priority_hints(const cldnn::device_info& info);

    #include "intel_gpu/runtime/options.inl"
};

}  // namespace ov::intel_gpu

namespace cldnn {
using ov::intel_gpu::ExecutionConfig;
}
