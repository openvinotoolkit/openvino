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
#include "cpu_map_scheduling.hpp"
#include "cpu_streams_calculation.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/system_conf.hpp"
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

static ov::RTMap get_cpu_rt_info(const ov::Model& model) {
    ov::RTMap rt_info;
    if (model.has_rt_info("intel_cpu_hints_config"))
        rt_info = model.get_rt_info<ov::AnyMap>("intel_cpu_hints_config");

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
    m_is_finalized = other.m_is_finalized;
    for (const auto& kv : other.m_options_map) {
        m_options_map.at(kv.first)->set_any(kv.second->get_any());
    }

    // m_stream_executor_config = other.m_stream_executor_config;
    m_model_prefer_threads = other.m_model_prefer_threads;
    m_stream_rank_table = other.m_stream_rank_table;
    m_stream_info_table = other.m_stream_info_table;
    m_num_sub_streams = other.m_num_sub_streams;
    m_proc_type_table = other.m_proc_type_table;
    m_numa_node_id = other.m_numa_node_id;
}

Config& Config::operator=(const Config& other) {
    m_user_properties = other.m_user_properties;
    m_is_finalized = other.m_is_finalized;
    for (const auto& kv : other.m_options_map) {
        m_options_map.at(kv.first)->set_any(kv.second->get_any());
    }

    // m_stream_executor_config = other.m_stream_executor_config;
    m_model_prefer_threads = other.m_model_prefer_threads;
    m_stream_rank_table = other.m_stream_rank_table;
    m_stream_info_table = other.m_stream_info_table;
    m_num_sub_streams = other.m_num_sub_streams;
    m_proc_type_table = other.m_proc_type_table;
    m_numa_node_id = other.m_numa_node_id;

    return *this;
}

Config Config::clone() const {
    Config new_config = *this;
    new_config.m_is_finalized = false;
    return new_config;
}


Config Config::clone(int num_sub_streamst) const {
    Config new_config = *this;
    new_config.m_num_sub_streams = num_sub_streamst;
    return new_config;
}

void Config::set_properties(const ov::AnyMap& config, OptionVisibility allowed_visibility) {
    const auto& it = config.find(ov::num_streams.name());
    if (it != config.end()) {
        auto num_streams = it->second.as<std::string>();
        auto new_config = config;
        new_config.at(ov::num_streams.name()) = num_streams;
        PluginConfig::set_user_property(new_config, allowed_visibility);

        return;
    }

    PluginConfig::set_user_property(config, allowed_visibility);
}

void Config::apply_rt_info(const IRemoteContext* context, const ov::RTMap& rt_info) {
    apply_rt_info_property(ov::hint::kv_cache_precision, rt_info);
    apply_rt_info_property(ov::hint::dynamic_quantization_group_size, rt_info);
    apply_rt_info_property(ov::key_cache_precision, rt_info);
    apply_rt_info_property(ov::value_cache_precision, rt_info);
    apply_rt_info_property(ov::key_cache_group_size, rt_info);
    apply_rt_info_property(ov::value_cache_group_size, rt_info);
}

void Config::apply_cpu_rt_info(const ov::RTMap& rt_info) {
    const auto model_prefer_name = std::string("m_model_prefer_threads");
    const auto rt_info_val = rt_info.find(model_prefer_name);
    if (rt_info_val != rt_info.end()) {
        try {
            m_model_prefer_threads = rt_info_val->second.as<int>();
        } catch (const ov::Exception&) {
            OPENVINO_THROW("Cache file doesn't have valid value for " + model_prefer_name);
        }
    }
}

void Config::finalize_impl(const IRemoteContext* context) {
    apply_hints();
    apply_threading_properties();

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

    if (!hasHardwareSupport(m_inference_precision) && m_inference_precision != ov::element::dynamic) {
        m_inference_precision = ov::element::f32;
    }

#if defined(__APPLE__)
    m_enable_cpu_reservation = false;
#endif
}

void Config::apply_model_specific_options(const IRemoteContext* context, const ov::Model& model) {
    apply_rt_info(context, get_rt_info(model));
    apply_cpu_rt_info(get_cpu_rt_info(model));

    if (!is_set_by_user(ov::intel_cpu::model_type)) {
        m_model_type = getModelType(model.shared_from_this());
    }

    if (-1 == m_model_prefer_threads) {
        m_model_prefer_threads = calc_model_prefer_threads(get_default_num_streams(), get_default_proc_type_table(), model.shared_from_this());
    }
}

void Config::apply_performance_hints() {
}

void Config::apply_threading_properties() {
    auto streams = get_default_num_streams();
    if (0 != streams || !is_set_by_user(ov::num_streams)) {
        m_proc_type_table = get_default_proc_type_table();
        m_stream_info_table = generate_stream_info(streams);
    }

    m_num_streams = ov::streams::Num(streams);
}

std::vector<std::vector<int>> Config::generate_stream_info(int streams) {
#if defined(__APPLE__)
    // CPUStreamExecutor doesn't support CPU reservation on Mac
    config.set_user_property(ov::hint::enable_cpu_reservation(false));
#endif

    if (m_proc_type_table.empty() || m_proc_type_table[0][ALL_PROC] == 0) {
        OPENVINO_THROW("m_proc_type_table is empty. No CPU resources available!");
    }

    m_proc_type_table = apply_scheduling_core_type(m_scheduling_core_type.value, m_proc_type_table);
    m_proc_type_table = apply_hyper_threading(m_enable_hyper_threading.value,
                                            is_set_by_user(ov::hint::enable_hyper_threading),
                                            ov::util::to_string(get_performance_mode()),
                                            m_proc_type_table);

    if (m_proc_type_table.size() > 1) {
        const auto cur_numa_node_id = m_numa_node_id < 0 ? get_current_numa_node_id() : m_numa_node_id;
        sort_table_by_numa_node_id(cur_numa_node_id, m_proc_type_table);
    }
    if (m_proc_type_table.empty() || m_proc_type_table[0][ALL_PROC] == 0) {
        OPENVINO_THROW("m_proc_type_table is empty. No valid CPU resources available!");
    }
    auto streams_info_table = get_streams_info_table(streams,
                                                     is_set_by_user(ov::num_streams) && get_num_streams() > 0,
                                                     get_inference_num_threads(),
                                                     get_num_requests(),
                                                     m_model_prefer_threads,
                                                     ov::util::to_string(get_performance_mode()),
                                                     get_model_distribution_policy(),
                                                     m_proc_type_table);
    if (streams_info_table.empty()) {
        OPENVINO_THROW("streams_info_table is empty!");
    }

    auto modelDistributionPolicy = get_model_distribution_policy();
    if (modelDistributionPolicy.find(ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL) != modelDistributionPolicy.end()) {
        m_stream_rank_table = get_streams_rank_table(streams_info_table, 1, m_num_sub_streams);
    }

    m_enable_cpu_pinning = check_cpu_pinning(get_enable_cpu_pinning(),
                                            is_set_by_user(ov::hint::enable_cpu_pinning),
                                            get_enable_cpu_reservation(),
                                            streams_info_table);

    return streams_info_table;
}

int Config::get_default_num_streams() {
#if defined(OV_CPU_WITH_SHL)
    // TODO: multi-stream execution is unsafe when SHL is used:
    //       The library uses global static variables as flags and counters.
    return 1;
#else
    // int streams_set
    auto streams = get_property(ov::num_streams.name()).as<ov::streams::Num>();
    if (get_exclusive_async_requests()) {
        return 1;
    } else if (streams == ov::streams::NUMA) {
        return ov::get_num_numa_nodes();
    } else if (streams == ov::streams::AUTO) {
        // bare minimum of streams (that evenly divides available number of cores)
        return ov::threading::IStreamsExecutor::Config::get_default_num_streams();
    }
#endif
    // if (is_set_by_user(ov::num_streams) && streams_set > 0) {
    //     streams = streams_set;
    // } else if (get_performance_mode() == ov::hint::PerformanceMode::LATENCY) {
    //     streams = 1;
    // } else if (get_performance_mode() == ov::hint::PerformanceMode::THROUGHPUT) {
    //     streams = 0;
    // } else {
    //     streams = streams_set == 1 ? 0 : streams_set;
    // }

    return streams.num;
}

std::vector<std::vector<int>> Config::get_default_proc_type_table() {
    std::lock_guard<std::mutex> lock{ov::threading::_streams_executor_mutex};
    return get_proc_type_table();
}

}  // namespace ov::intel_cpu
