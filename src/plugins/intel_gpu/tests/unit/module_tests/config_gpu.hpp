// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "config_common.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "config_gpu_debug_properties.hpp"
#include <thread>

namespace ov {
namespace intel_gpu {

struct NewExecutionConfig : public ov::PluginConfig {
    NewExecutionConfig();

    #define OV_CONFIG_OPTION(PropertyNamespace, PropertyVar, ...) \
        ConfigOption<decltype(PropertyNamespace::PropertyVar)::value_type> PropertyVar = \
            ConfigOption<decltype(PropertyNamespace::PropertyVar)::value_type>(GET_EXCEPT_LAST(__VA_ARGS__));


    #include "config_gpu_options.inl"
    #include "config_gpu_debug_options.inl"

    #undef OV_CONFIG_OPTION

    void finalize_impl(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) override;

protected:
    // Note that RT info property value has lower priority than values set by user via core.set_property or passed to compile_model call
    // So this method should be called after setting all user properties, but before apply_user_properties() call.
    void apply_rt_info(const cldnn::device_info& info, const ov::RTMap& rt_info);

    void apply_user_properties(const cldnn::device_info& info);
    void apply_hints(const cldnn::device_info& info);
    void apply_execution_hints(const cldnn::device_info& info);
    void apply_performance_hints(const cldnn::device_info& info);
    void apply_priority_hints(const cldnn::device_info& info);
    void apply_debug_options(const cldnn::device_info& info);
};


}  // namespace intel_gpu
}  // namespace ov
