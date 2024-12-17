// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config_gpu.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "config_gpu_debug_properties.hpp"


namespace ov {
namespace intel_gpu {

NewExecutionConfig::NewExecutionConfig() : ov::PluginConfig() {
    #define OV_CONFIG_OPTION(PropertyNamespace, PropertyVar, ...) \
        m_options_map[PropertyNamespace::PropertyVar.name()] = &PropertyVar;

    OV_CONFIG_OPTION(ov, enable_profiling, false, "Enable profiling for the plugin")
    #include "config_gpu_options.inl"
    #include "config_gpu_debug_options.inl"

    #undef OV_CONFIG_OPTION
}

void NewExecutionConfig::finalize_impl(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) {
    const auto& device_info = std::dynamic_pointer_cast<RemoteContextImpl>(context)->get_engine().get_device_info();
    apply_user_properties(device_info);
    apply_rt_info(device_info, rt_info);
}

void NewExecutionConfig::apply_execution_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::execution_mode)) {
        const auto mode = get_property(ov::hint::execution_mode);
        if (!is_set_by_user(ov::hint::inference_precision)) {
            if (mode == ov::hint::ExecutionMode::ACCURACY) {
                set_property(ov::hint::inference_precision(ov::element::undefined));
            } else if (mode == ov::hint::ExecutionMode::PERFORMANCE) {
                if (info.supports_fp16)
                    set_property(ov::hint::inference_precision(ov::element::f16));
                else
                    set_property(ov::hint::inference_precision(ov::element::f32));
            }
        }
    }
}

void NewExecutionConfig::apply_performance_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::performance_mode)) {
        const auto mode = get_property(ov::hint::performance_mode);
        if (!is_set_by_user(ov::num_streams)) {
            if (mode == ov::hint::PerformanceMode::LATENCY) {
                set_property(ov::num_streams(1));
            } else if (mode == ov::hint::PerformanceMode::THROUGHPUT) {
                set_property(ov::num_streams(ov::streams::AUTO));
            }
        }
    }

    if (get_property(ov::num_streams) == ov::streams::AUTO) {
        int32_t n_streams = std::max<int32_t>(info.num_ccs, 2);
        set_property(ov::num_streams(n_streams));
    }

    if (get_property(ov::internal::exclusive_async_requests)) {
        set_property(ov::num_streams(1));
    }

    // Allow kernels reuse only for single-stream scenarios
    if (get_property(ov::intel_gpu::hint::enable_kernels_reuse)) {
        if (get_property(ov::num_streams) != 1) {
            set_property(ov::intel_gpu::hint::enable_kernels_reuse(false));
        }
    }
}

void NewExecutionConfig::apply_priority_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::model_priority)) {
        const auto priority = get_property(ov::hint::model_priority);
        if (!is_set_by_user(ov::intel_gpu::hint::queue_priority)) {
            set_property(ov::intel_gpu::hint::queue_priority(priority));
        }
    }
}

void NewExecutionConfig::apply_debug_options(const cldnn::device_info& info) {
    // GPU_DEBUG_GET_INSTANCE(debug_config);
    // GPU_DEBUG_IF(!debug_config->dump_graphs.empty()) {
    //     set_property(ov::intel_gpu::dump_graphs(debug_config->dump_graphs));
    // }

    // GPU_DEBUG_IF(debug_config->serialize_compile == 1) {
    //     set_property(ov::compilation_num_threads(1));
    // }

    // GPU_DEBUG_IF(!debug_config->dump_profiling_data.empty()) {
    //     GPU_DEBUG_COUT << "[WARNING] ov::enable_profiling property was forced because of enabled OV_GPU_DumpProfilingData debug option\n";
    //     set_property(ov::enable_profiling(true));
    // }

    // GPU_DEBUG_IF(debug_config->disable_dynamic_impl == 1) {
    //     set_property(ov::intel_gpu::use_only_static_kernels_for_dynamic_shape(true));
    // }

    // GPU_DEBUG_IF(debug_config->dynamic_quantize_group_size != debug_config->DYNAMIC_QUANTIZE_GROUP_SIZE_NOT_SET) {
    //     if (debug_config->dynamic_quantize_group_size == -1)
    //         set_property(ov::hint::dynamic_quantization_group_size(UINT64_MAX));
    //     else
    //         set_property(ov::hint::dynamic_quantization_group_size(debug_config->dynamic_quantize_group_size));
    // }

    // GPU_DEBUG_IF(debug_config->use_kv_cache_compression != -1) {
    //     GPU_DEBUG_IF(debug_config->use_kv_cache_compression == 1) {
    //         set_property(ov::hint::kv_cache_precision(ov::element::i8));
    //     } else {
    //         set_property(ov::hint::kv_cache_precision(ov::element::undefined));
    //     }
    // }
}

void NewExecutionConfig::apply_hints(const cldnn::device_info& info) {
    apply_execution_hints(info);
    apply_performance_hints(info);
    apply_priority_hints(info);
    apply_debug_options(info);
}

void NewExecutionConfig::apply_user_properties(const cldnn::device_info& info) {
    apply_hints(info);
    if (!is_set_by_user(ov::intel_gpu::enable_lp_transformations)) {
        set_property(ov::intel_gpu::enable_lp_transformations(info.supports_imad || info.supports_immad));
    }
    if (info.supports_immad) {
        set_property(ov::intel_gpu::use_onednn(true));
    }
    if (get_property(ov::intel_gpu::use_onednn)) {
        set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    }

    // Enable KV-cache compression by default for non-systolic platforms
    if (!is_set_by_user(ov::hint::kv_cache_precision) && !info.supports_immad) {
        set_property(ov::hint::kv_cache_precision(ov::element::i8));
    }

    // Enable dynamic quantization by default for non-systolic platforms
    if (!is_set_by_user(ov::hint::dynamic_quantization_group_size) && !info.supports_immad) {
        set_property(ov::hint::dynamic_quantization_group_size(32));
    }
}

void NewExecutionConfig::apply_rt_info(const cldnn::device_info& info, const ov::RTMap& rt_info) {
    if (!info.supports_immad) {
        apply_rt_info_property(ov::hint::kv_cache_precision, rt_info);
        apply_rt_info_property(ov::hint::activations_scale_factor, rt_info);
    }
    apply_rt_info_property(ov::hint::dynamic_quantization_group_size, rt_info);
}

}  // namespace intel_gpu
}  // namespace ov
