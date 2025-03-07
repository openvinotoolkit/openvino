// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.h"
#include "utils/precision_support.h"
#include "utils/codec_xor.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/paged_attention.hpp"

namespace ov::intel_cpu {

static ov::RTMap get_rt_info(const ov::Model& model) {
    ov::RTMap rt_info;
    if (model.has_rt_info("runtime_options"))
        rt_info = model.get_rt_info<ov::AnyMap>("runtime_options");

    if (model.has_rt_info("__weights_path")) {
        rt_info[ov::weights_path.name()] = model.get_rt_info<ov::Any>("__weights_path");
    }
    return rt_info;
}

static ModelType getModelType(const std::shared_ptr<const Model>& model) {
    if (op::util::has_op_with_type<op::v1::Convolution>(model) ||
        op::util::has_op_with_type<op::v1::ConvolutionBackpropData>(model)) {
        return ModelType::CNN;
    }

    if ((op::util::has_op_with_type<op::v13::ScaledDotProductAttention>(model) && model->get_variables().size() > 0) ||
        op::util::has_op_with_type<ov::op::PagedAttentionExtension>(model)) {
        return ModelType::LLM;
    }

    return ModelType::UNKNOWN;
}

Config::Config() : ov::PluginConfig() {
    set_default_values();
}

Config::Config(const Config& other) : Config() {
    m_user_properties = other.m_user_properties;
    m_is_finalized = false; // copy is not automatically finalized
    for (const auto& kv : other.m_options_map) {
        m_options_map.at(kv.first)->set_any(kv.second->get_any());
    }

    streamExecutorConfig = other.streamExecutorConfig;
    streamsRankTable = other.streamsRankTable;
    streamsRankLevel = other.streamsRankLevel;
    numSubStreams = other.numSubStreams;
    enableNodeSplit = other.enableNodeSplit;
}

Config& Config::operator=(const Config& other) {
    m_user_properties = other.m_user_properties;
    m_is_finalized = false; // copy is not automatically finalized
    for (const auto& kv : other.m_options_map) {
        m_options_map.at(kv.first)->set_any(kv.second->get_any());
    }

    streamExecutorConfig = other.streamExecutorConfig;
    streamsRankTable = other.streamsRankTable;
    streamsRankLevel = other.streamsRankLevel;
    numSubStreams = other.numSubStreams;
    enableNodeSplit = other.enableNodeSplit;

    return *this;
}

void Config::apply_rt_info(const IRemoteContext* context, const ov::RTMap& rt_info) {
    apply_rt_info_property(ov::hint::kv_cache_precision, rt_info);
    apply_rt_info_property(ov::hint::dynamic_quantization_group_size, rt_info);
    apply_rt_info_property(ov::key_cache_precision, rt_info);
    apply_rt_info_property(ov::value_cache_precision, rt_info);
    apply_rt_info_property(ov::key_cache_group_size, rt_info);
    apply_rt_info_property(ov::value_cache_group_size, rt_info);
}

void Config::finalize_impl(const IRemoteContext* context) {
    apply_hints();

    if (get_exclusive_async_requests()) {
        m_num_streams = 1;
    }

#if defined(OV_CPU_WITH_SHL)
    // TODO: multi-stream execution is unsafe when SHL is used:
    //       The library uses global static variables as flags and counters.
    m_num_streams = 1;
#endif

    if (!m_cache_encryption_callbacks.value.encrypt || !m_cache_encryption_callbacks.value.decrypt) {
        m_cache_encryption_callbacks.value.encrypt = codec_xor_str;
        m_cache_encryption_callbacks.value.decrypt = codec_xor_str;
    }
}

void Config::set_default_values() {
#if defined(OPENVINO_ARCH_X86_64)
    m_cpu_runtime_cache_capacity = 5000ul;
#else
    // TODO: Executor cache may leads to incorrect behavior on oneDNN ACL primitives
    // TODO: Verify on RISC-V platforms
    m_cpu_runtime_cache_capacity = 0ul;
#endif

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    m_enable_lp_transformations = true;
#else
    // Currently INT8 mode is not optimized on ARM / RISCV or other non-x86 platforms, fallback to FP32 mode.
    m_enable_lp_transformations = false;
#endif
}

void Config::apply_hints() {
    apply_execution_hints();
    apply_performance_hints();
}

void Config::apply_execution_hints() {
    if (get_execution_mode() == ov::hint::ExecutionMode::PERFORMANCE) {
        if (!is_set_by_user(ov::hint::inference_precision)) {
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
            m_inference_precision = ov::element::f16;
#else
            if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16))
                m_inference_precision = ov::element::bf16;
#endif
        }
        if (!is_set_by_user(ov::hint::dynamic_quantization_group_size)) {
            m_dynamic_quantization_group_size = 32;
        }
        if (!is_set_by_user(ov::hint::kv_cache_precision)) {
#if defined(OPENVINO_ARCH_X86_64)
            m_kv_cache_precision = ov::element::u8;
#else
            m_kv_cache_precision = ov::element::f16;
#endif
        }
#if defined(OV_CPU_WITH_ACL)
        if (!is_set_by_user(ov::intel_cpu::acl_fast_math)) {
            m_acl_fast_math = true;
        }
#endif
    }

    if (get_execution_mode() == ov::hint::ExecutionMode::ACCURACY) {
        if (!is_set_by_user(ov::hint::inference_precision)) {
            m_inference_precision = ov::element::dynamic;
        }
        if (!is_set_by_user(ov::hint::dynamic_quantization_group_size)) {
            m_dynamic_quantization_group_size = 0;
        }
        if (!is_set_by_user(ov::hint::kv_cache_precision)) {
            m_kv_cache_precision = ov::element::f32;
        }
#if defined(OV_CPU_WITH_ACL)
        if (!is_set_by_user(ov::intel_cpu::acl_fast_math)) {
            m_acl_fast_math = false;
        }
#endif
    }

    // key/value cache precision are low-level options, so has higher priority than m_kv_cache_precision
    if (!is_set_by_user(ov::key_cache_precision)) {
        m_key_cache_precision = m_kv_cache_precision;
    }
    if (!is_set_by_user(ov::value_cache_precision)) {
        m_value_cache_precision = m_kv_cache_precision;
    }

    if (!hasHardwareSupport(m_inference_precision)) {
        m_inference_precision = ov::element::f32;
    }



#if defined(__APPLE__)
    m_enable_cpu_reservation = false;
#endif
}

void Config::apply_model_specific_options(const IRemoteContext* context, const ov::Model& model) {
    apply_rt_info(context, get_rt_info(model));

    if (!is_set_by_user(ov::intel_cpu::model_type)) {
        m_model_type = getModelType(model.shared_from_this());
    }
}

void Config::apply_performance_hints() {
    // if (is_set_by_user(ov::hint::performance_mode)) {
    //     const auto mode = get_property(ov::hint::performance_mode);
    //     if (!is_set_by_user(ov::num_streams)) {
    //         if (mode == ov::hint::PerformanceMode::LATENCY) {
    //             set_property(ov::num_streams(1));
    //         } else if (mode == ov::hint::PerformanceMode::THROUGHPUT) {
    //             set_property(ov::num_streams(ov::streams::AUTO));
    //         }
    //     }
    // }

    // if (get_property(ov::num_streams) == ov::streams::AUTO) {
    //     int32_t n_streams = std::max<int32_t>(info.num_ccs, 2);
    //     set_property(ov::num_streams(n_streams));
    // }

    // if (get_property(ov::internal::exclusive_async_requests)) {
    //     set_property(ov::num_streams(1));
    // }

    // // Allow kernels reuse only for single-stream scenarios
    // if (get_property(ov::intel_gpu::hint::enable_kernels_reuse)) {
    //     if (get_property(ov::num_streams) != 1) {
    //         set_property(ov::intel_gpu::hint::enable_kernels_reuse(false));
    //     }
    // }
}

}  // namespace ov::intel_cpu
