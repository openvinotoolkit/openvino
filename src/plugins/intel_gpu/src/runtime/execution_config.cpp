// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/runtime/plugin_config.hpp"
#include "openvino/runtime/properties.hpp"


namespace ov {
namespace intel_gpu {

ExecutionConfig::ExecutionConfig() : ov::PluginConfig() {
    #define OV_CONFIG_OPTION(...) OV_CONFIG_OPTION_MAPPING(__VA_ARGS__)
    #include "intel_gpu/runtime/options.inl"
    #undef OV_CONFIG_OPTION
}

ExecutionConfig::ExecutionConfig(const ExecutionConfig& other) : ExecutionConfig() {
    m_user_properties = other.m_user_properties;
    m_is_finalized = false; // copy is not automatically finalized
    for (const auto& kv : other.m_options_map) {
        m_options_map.at(kv.first)->set_any(kv.second->get_any());
    }
}

ExecutionConfig& ExecutionConfig::operator=(const ExecutionConfig& other) {
    m_user_properties = other.m_user_properties;
    m_is_finalized = false; // copy is not automatically finalized
    for (const auto& kv : other.m_options_map) {
        m_options_map.at(kv.first)->set_any(kv.second->get_any());
    }
    return *this;
}

void ExecutionConfig::finalize(cldnn::engine& engine) {
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    PluginConfig::finalize(ctx, {});
}

void ExecutionConfig::apply_rt_info(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) {
    const auto& info = std::dynamic_pointer_cast<RemoteContextImpl>(context)->get_engine().get_device_info();
    if (!info.supports_immad) {
        apply_rt_info_property(ov::hint::kv_cache_precision, rt_info);
        apply_rt_info_property(ov::hint::activations_scale_factor, rt_info);
    }
    apply_rt_info_property(ov::hint::dynamic_quantization_group_size, rt_info);

    // WEIGHTS_PATH is used for the weightless cache mechanism which is used only with
    // ov::CacheMode::OPTIMIZE_SIZE setting. Not setting WEIGHTS_PATH will result in not
    // using that mechanism.
    if (get_cache_mode() == ov::CacheMode::OPTIMIZE_SIZE) {
        apply_rt_info_property(ov::weights_path, rt_info);
    }
}

void ExecutionConfig::finalize_impl(std::shared_ptr<IRemoteContext> context) {
    if (m_help) {
        print_help();
        exit(-1);
    }

    const auto& info = std::dynamic_pointer_cast<RemoteContextImpl>(context)->get_engine().get_device_info();
    apply_hints(info);
    if (!is_set_by_user(ov::intel_gpu::enable_lp_transformations)) {
        m_enable_lp_transformations = info.supports_imad || info.supports_immad;
    }
    if (info.supports_immad) {
        m_use_onednn = true;
    }
    if (get_use_onednn()) {
        m_queue_type = QueueTypes::in_order;
    }

    // Enable KV-cache compression by default for non-systolic platforms
    if (!is_set_by_user(ov::hint::kv_cache_precision) && !info.supports_immad) {
        m_kv_cache_precision = ov::element::i8;
    }

    // Enable dynamic quantization by default for non-systolic platforms
    if (!is_set_by_user(ov::hint::dynamic_quantization_group_size) && !info.supports_immad) {
        m_dynamic_quantization_group_size = 32;
    }

    if (!get_force_implementations().empty()) {
        m_optimize_data = true;
    }
}

void ExecutionConfig::apply_hints(const cldnn::device_info& info) {
    apply_execution_hints(info);
    apply_performance_hints(info);
    apply_priority_hints(info);
}

void ExecutionConfig::apply_execution_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::execution_mode)) {
        const auto mode = get_execution_mode();
        if (!is_set_by_user(ov::hint::inference_precision)) {
            if (mode == ov::hint::ExecutionMode::ACCURACY) {
                m_inference_precision = ov::element::undefined;
            } else if (mode == ov::hint::ExecutionMode::PERFORMANCE) {
                if (info.supports_fp16)
                    m_inference_precision = ov::element::f16;
                else
                    m_inference_precision = ov::element::f32;
            }
        }
    }
}

void ExecutionConfig::apply_performance_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::performance_mode)) {
        const auto mode = get_performance_mode();
        if (!is_set_by_user(ov::num_streams)) {
            if (mode == ov::hint::PerformanceMode::LATENCY) {
                m_num_streams = 1;
            } else if (mode == ov::hint::PerformanceMode::THROUGHPUT) {
                m_num_streams = ov::streams::AUTO;
            }
        }
    }

    if (get_num_streams() == ov::streams::AUTO) {
        int32_t n_streams = std::max<int32_t>(info.num_ccs, 2);
        m_num_streams = n_streams;
    }

    if (get_exclusive_async_requests()) {
        m_num_streams = 1;
    }

    // Allow kernels reuse only for single-stream scenarios
    if (get_enable_kernels_reuse()) {
        if (get_num_streams() != 1) {
            m_enable_kernels_reuse = false;
        }
    }
}

void ExecutionConfig::apply_priority_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::model_priority)) {
        const auto priority = get_model_priority();
        if (!is_set_by_user(ov::intel_gpu::hint::queue_priority)) {
            m_queue_priority = priority;
        }
    }
}

const ov::PluginConfig::OptionsDesc& ExecutionConfig::get_options_desc() const {
    static  ov::PluginConfig::OptionsDesc help_map {
        #define OV_CONFIG_OPTION(...) OV_CONFIG_OPTION_HELP(__VA_ARGS__)
        #include "intel_gpu/runtime/options.inl"
        #undef OV_CONFIG_OPTION
    };
    return help_map;
}

}  // namespace intel_gpu
}  // namespace ov
