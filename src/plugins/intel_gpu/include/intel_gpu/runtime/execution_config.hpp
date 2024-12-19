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

struct ExecutionConfig : public ov::PluginConfig {
    ExecutionConfig();
    ExecutionConfig(std::initializer_list<ov::AnyMap::value_type> values) : ExecutionConfig() { set_property(ov::AnyMap(values)); }
    explicit ExecutionConfig(const ov::AnyMap& properties) : ExecutionConfig() { set_property(properties); }
    explicit ExecutionConfig(const ov::AnyMap::value_type& property) : ExecutionConfig() { set_property(property); }

    ExecutionConfig(const ExecutionConfig& other);
    ExecutionConfig& operator=(const ExecutionConfig& other);

    #define OV_CONFIG_OPTION(...) OV_CONFIG_DECLARE_OPTION(__VA_ARGS__)
    #include "intel_gpu/runtime/options.inl"
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

namespace cldnn {
using ov::intel_gpu::ExecutionConfig;
}
