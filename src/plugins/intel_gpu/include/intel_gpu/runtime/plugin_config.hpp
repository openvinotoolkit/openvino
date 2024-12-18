// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/plugin_config.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include <thread>

namespace ov {
namespace intel_gpu {

struct NewExecutionConfig : public ov::PluginConfig {
    NewExecutionConfig();
    NewExecutionConfig(std::initializer_list<ov::AnyMap::value_type> values) : NewExecutionConfig() { set_property(ov::AnyMap(values)); }
    explicit NewExecutionConfig(const ov::AnyMap& properties) : NewExecutionConfig() { set_property(properties); }
    explicit NewExecutionConfig(const ov::AnyMap::value_type& property) : NewExecutionConfig() { set_property(property); }

    NewExecutionConfig(const NewExecutionConfig& other);
    NewExecutionConfig& operator=(const NewExecutionConfig& other);

    #define OV_CONFIG_OPTION(PropertyNamespace, PropertyVar, ...) \
        ConfigOption<decltype(PropertyNamespace::PropertyVar)::value_type> m_ ## PropertyVar = \
            ConfigOption<decltype(PropertyNamespace::PropertyVar)::value_type>(GET_EXCEPT_LAST(__VA_ARGS__));

    #include "options_release.inl"
    #include "options_debug.inl"

    #undef OV_CONFIG_OPTION

    void finalize_impl(std::shared_ptr<IRemoteContext> context) override;
    void apply_rt_info(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) override;

private:
    void apply_user_properties(const cldnn::device_info& info);
    void apply_hints(const cldnn::device_info& info);
    void apply_execution_hints(const cldnn::device_info& info);
    void apply_performance_hints(const cldnn::device_info& info);
    void apply_priority_hints(const cldnn::device_info& info);
};

}  // namespace intel_gpu
}  // namespace ov
