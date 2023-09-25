// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.h"

#include <string>
#include <map>
#include <algorithm>

#include "ie_plugin_config.hpp"
#include "cpu/cpu_config.hpp"
#include "ie_common.h"
#include "ie_parallel.hpp"
#include "ie_system_conf.h"

#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils/debug_capabilities.h"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov {
namespace intel_cpu {

using namespace ov::threading;
using namespace dnnl::impl::cpu::x64;

Config::Config() {
    // this is default mode
#if defined(__APPLE__) || defined(_WIN32)
    streamExecutorConfig._threadBindingType = IStreamsExecutor::NONE;
#else
    streamExecutorConfig._threadBindingType = IStreamsExecutor::CORES;
#endif

// for the TBB code-path, additional configuration depending on the OS and CPU types
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
#    if defined(__APPLE__) || defined(_WIN32)
    // 'CORES' is not implemented for Win/MacOS; so the 'NONE' or 'NUMA' is default
    auto numaNodes = get_available_numa_nodes();
    if (numaNodes.size() > 1) {
        streamExecutorConfig._threadBindingType = IStreamsExecutor::NUMA;
    } else {
        streamExecutorConfig._threadBindingType = IStreamsExecutor::NONE;
    }
#    endif

    if (get_available_cores_types().size() > 1 /*Hybrid CPU*/) {
        streamExecutorConfig._threadBindingType = IStreamsExecutor::HYBRID_AWARE;
    }
#endif

    CPU_DEBUG_CAP_ENABLE(applyDebugCapsProperties());

    updateProperties();
}

#ifdef CPU_DEBUG_CAPS
/**
 * Debug capabilities configuration has more priority than common one
 * Some of the debug capabilities also require to enable some of common
 * configuration properties
 */
void Config::applyDebugCapsProperties() {
    // always enable perf counters for verbose mode and performance summary
    if (!debugCaps.verbose.empty() || !debugCaps.summaryPerf.empty())
        collectPerfCounters = true;
}
#endif

void Config::readProperties(const ov::AnyMap& prop, const ModelType modelType) {
    const auto streamExecutorConfigKeys =
        streamExecutorConfig.get_property(ov::supported_properties.name()).as<std::vector<std::string>>();
    const auto hintsConfigKeys = perfHintsConfig.SupportedKeys();
    for (const auto& kvp : prop) {
        const auto& key = kvp.first;
        const auto& val = kvp.second.as<std::string>();
        IE_SUPPRESS_DEPRECATED_START
        if (streamExecutorConfigKeys.end() !=
            std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key)) {
            streamExecutorConfig.set_property(key, val);
            if (key == ov::affinity.name()) {
                const auto affinity_val = ov::util::from_string(val, ov::affinity);
                if (affinity_val == ov::Affinity::CORE || affinity_val == ov::Affinity::HYBRID_AWARE) {
                    enableCpuPinning = true;
                    changedCpuPinning = true;
                } else if (affinity_val == ov::Affinity::NUMA) {
                    enableCpuPinning = false;
                    changedCpuPinning = true;
                }
            }
        } else if (hintsConfigKeys.end() != std::find(hintsConfigKeys.begin(), hintsConfigKeys.end(), key)) {
            perfHintsConfig.SetConfig(key, val);
        } else if (key == ov::hint::enable_cpu_pinning.name()) {
            if (val == Config::YES) {
                enableCpuPinning = true;
                changedCpuPinning = true;
            } else if (val == Config::NO) {
                enableCpuPinning = false;
                changedCpuPinning = true;
            } else {
                OPENVINO_THROW("Wrong value ",
                               val,
                               "for property key ",
                               ov::hint::enable_cpu_pinning.name(),
                               ". Expected only true/false.");
            }
        } else if (key == ov::hint::scheduling_core_type.name()) {
            const auto core_type = ov::util::from_string(val, ov::hint::scheduling_core_type);
            if (core_type == ov::hint::SchedulingCoreType::ANY_CORE ||
                core_type == ov::hint::SchedulingCoreType::PCORE_ONLY ||
                core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) {
                schedulingCoreType = core_type;
            } else {
                OPENVINO_THROW("Wrong value ",
                               val,
                               "for property key ",
                               ov::hint::scheduling_core_type.name(),
                               ". Expected only ",
                               ov::hint::SchedulingCoreType::ANY_CORE,
                               "/",
                               ov::hint::SchedulingCoreType::PCORE_ONLY,
                               "/",
                               ov::hint::SchedulingCoreType::ECORE_ONLY);
            }
        } else if (key == ov::hint::enable_hyper_threading.name()) {
            if (val == Config::YES) {
                enableHyperThreading = true;
                changedHyperThreading = true;
            } else if (val == Config::NO) {
                enableHyperThreading = false;
                changedHyperThreading = true;
            } else {
                OPENVINO_THROW("Wrong value ",
                               val,
                               "for property key ",
                               ov::hint::enable_hyper_threading.name(),
                               ". Expected only true/false.");
            }
        } else if (key == ov::intel_cpu::sparse_weights_decompression_rate.name()) {
            float val_f = 0.0f;
            try {
                val_f = std::stof(val);
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value for property key ",
                               ov::intel_cpu::sparse_weights_decompression_rate.name(),
                               ". Expected only float numbers");
            }
            if (val_f < 0.f || val_f > 1.f) {
                OPENVINO_THROW("Wrong value for property key ",
                               ov::intel_cpu::sparse_weights_decompression_rate.name(),
                               ". Sparse rate must be in range [0.0f,1.0f]");
            } else {
                fcSparseWeiDecompressionRate = val_f;
            }
        } else if (key == ov::enable_profiling.name()) {
            if (val == Config::YES)
                collectPerfCounters = true;
            else if (val == Config::NO)
                collectPerfCounters = false;
            else
                OPENVINO_THROW("Wrong value for property key ", ov::enable_profiling.name(), ". Expected only YES/NO");
        } else if (key == ov::exclusive_async_requests.name()) {
            if (val == Config::YES)
                exclusiveAsyncRequests = true;
            else if (val == Config::NO)
                exclusiveAsyncRequests = false;
            else
                OPENVINO_THROW("Wrong value for property key ",
                               ov::exclusive_async_requests.name(),
                               ". Expected only YES/NO");
            IE_SUPPRESS_DEPRECATED_START
        } else if (key.compare(InferenceEngine::PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT) == 0) {
            IE_SUPPRESS_DEPRECATED_END
            // empty string means that dumping is switched off
            dumpToDot = val;
        } else if (key.compare(ov::internal::lp_transforms_mode.name()) == 0) {
            if (val == Config::NO)
                lpTransformsMode = LPTransformsMode::Off;
            else if (val == Config::YES)
                lpTransformsMode = LPTransformsMode::On;
            else
                OPENVINO_THROW("Wrong value for property key ", ov::internal::lp_transforms_mode.name());
        } else if (key == ov::device::id.name()) {
            device_id = val;
            if (!device_id.empty()) {
                OPENVINO_THROW("CPU plugin supports only '' as device id");
            }
        } else if (key == ov::enforce_bf16.name()) {
            if (val == Config::YES) {
                if (mayiuse(avx512_core)) {
                    inferencePrecision = ov::element::bf16;
                } else {
                    OPENVINO_THROW("Platform doesn't support BF16 format");
                }
            } else if (val == Config::NO) {
                inferencePrecision = ov::element::f32;
            } else {
                OPENVINO_THROW("Wrong value for property key ", ov::enforce_bf16.name(), ". Expected only YES/NO");
            }
            inferencePrecisionSetExplicitly = true;
        } else if (key == ov::hint::inference_precision.name()) {
            if (val == "bf16") {
                if (mayiuse(avx512_core)) {
                    inferencePrecision = ov::element::bf16;
                    inferencePrecisionSetExplicitly = true;
                }
            } else if (val == "f16") {
#if defined(OPENVINO_ARCH_X86_64)
                if (mayiuse(avx512_core_fp16) || mayiuse(avx512_core_amx_fp16)) {
                    inferencePrecision = ov::element::f16;
                    inferencePrecisionSetExplicitly = true;
                }
#elif defined(OV_CPU_ARM_ENABLE_FP16)
// TODO: add runtime FP16 feature support check for ARM
                inferencePrecision = ov::element::f16;
                inferencePrecisionSetExplicitly = true;
#endif
            } else if (val == "f32") {
                inferencePrecision = ov::element::f32;
                inferencePrecisionSetExplicitly = true;
            } else {
                OPENVINO_THROW("Wrong value for property key ",
                               ov::hint::inference_precision.name(),
                               ". Supported values: bf16, f32");
            }
        } else if (ov::intel_cpu::cpu_runtime_cache_capacity.name() == key) {
            int val_i = -1;
            try {
                val_i = std::stoi(val);
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value for property key ",
                               ov::intel_cpu::cpu_runtime_cache_capacity.name(),
                               ". Expected only integer numbers");
            }
            // any negative value will be treated
            // as zero that means disabling the cache
            rtCacheCapacity = std::max(val_i, 0);
        } else if (ov::intel_cpu::denormals_optimization.name() == key) {
            if (val == Config::YES) {
                denormalsOptMode = DenormalsOptMode::DO_On;
            } else if (val == Config::NO) {
                denormalsOptMode = DenormalsOptMode::DO_Off;
            } else {
                denormalsOptMode = DenormalsOptMode::DO_Keep;
                OPENVINO_THROW("Wrong value for property key ",
                               ov::intel_cpu::denormals_optimization.name(),
                               ". Expected only YES/NO");
            }
        } else if (key == ov::snippets_mode.name()) {
            if (val == ov::util::to_string(ov::SnippetsMode::ENABLE))
                snippetsMode = SnippetsMode::Enable;
            else if (val == ov::util::to_string(ov::SnippetsMode::IGNORE_CALLBACK))
                snippetsMode = SnippetsMode::IgnoreCallback;
            else if (val == ov::util::to_string(ov::SnippetsMode::DISABLE))
                snippetsMode = SnippetsMode::Disable;
            else
                OPENVINO_THROW("Wrong value for property key ",
                               ov::snippets_mode.name(),
                               ". Expected values: ENABLE/DISABLE/IGNORE_CALLBACK");
        } else if (key == ov::hint::execution_mode.name()) {
            if (val == "PERFORMANCE") {
                executionMode = ov::hint::ExecutionMode::PERFORMANCE;
            } else if (val == "ACCURACY") {
                executionMode = ov::hint::ExecutionMode::ACCURACY;
            } else {
                OPENVINO_THROW("Wrong value for property key ",
                               ov::hint::execution_mode.name(),
                               ". Supported values: PERFORMANCE, ACCURACY");
            }
        } else {
            IE_THROW(NotFound) << "Unsupported property " << key << " by CPU plugin";
            // OPENVINO_THROW_HELPER(NotFound, "Unsupported property ", key, " by CPU plugin");
        }
        IE_SUPPRESS_DEPRECATED_END
    }
    // apply execution mode after all the params are handled to prevent possible conflicts
    // when both execution_mode and inference_precision are specified
    if (!inferencePrecisionSetExplicitly) {
        if (executionMode == ov::hint::ExecutionMode::PERFORMANCE) {
            if (mayiuse(avx512_core_bf16))
                inferencePrecision = ov::element::bf16;
            else
                inferencePrecision = ov::element::f32;
        } else {
            inferencePrecision = ov::element::f32;
        }
    }

    if (!prop.empty())
        _config.clear();

    if (exclusiveAsyncRequests) {  // Exclusive request feature disables the streams
        streamExecutorConfig._streams = 1;
        streamExecutorConfig._streams_changed = true;
    }

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    // TODO: multi-stream execution has functional issues on ARM target
    streamExecutorConfig._streams = 1;
    streamExecutorConfig._streams_changed = true;
#endif
    this->modelType = modelType;

    CPU_DEBUG_CAP_ENABLE(applyDebugCapsProperties());
    updateProperties();
}

void Config::updateProperties() {
    if (!_config.empty())
        return;

    switch (streamExecutorConfig._threadBindingType) {
    case IStreamsExecutor::ThreadBindingType::NONE:
        _config.insert({ov::internal::cpu_bind_thread.name(), std::string(Config::NO)});
        break;
    case IStreamsExecutor::ThreadBindingType::CORES:
        _config.insert({ov::internal::cpu_bind_thread.name(), std::string(Config::YES)});
        break;
    case IStreamsExecutor::ThreadBindingType::NUMA:
        _config.insert(
            {ov::internal::cpu_bind_thread.name(), ov::util::to_string(IStreamsExecutor::ThreadBindingType::NUMA)});
        break;
    case IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
        _config.insert({ov::internal::cpu_bind_thread.name(),
                        ov::util::to_string(IStreamsExecutor::ThreadBindingType::HYBRID_AWARE)});
        break;
    }
    if (collectPerfCounters == true)
        _config.insert({ov::perf_count.name(), std::string(Config::YES)});
    else
        _config.insert({ov::perf_count.name(), std::string(Config::NO)});
    if (exclusiveAsyncRequests == true)
        _config.insert({ov::internal::exclusive_async_requests.name(), std::string(Config::YES)});
    else
        _config.insert({ov::internal::exclusive_async_requests.name(), std::string(Config::NO)});

    _config.insert({ov::device::id.name(), device_id});

    _config.insert({ov::num_streams.name(), std::to_string(streamExecutorConfig._streams)});
    _config.insert({ov::inference_num_threads.name(), std::to_string(streamExecutorConfig._threads)});
    IE_SUPPRESS_DEPRECATED_START
    _config.insert({InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                    std::to_string(streamExecutorConfig._streams)});
    _config.insert(
        {InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, std::to_string(streamExecutorConfig._threads)});
    _config.insert({InferenceEngine::PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT, dumpToDot});
    IE_SUPPRESS_DEPRECATED_END;
    if (inferencePrecision == ov::element::bf16) {
        _config.insert({ov::enforce_bf16.name(), std::string(Config::YES)});
    } else {
        _config.insert({ov::enforce_bf16.name(), std::string(Config::NO)});
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    _config.insert({InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, perfHintsConfig.ovPerfHint});
    _config.insert({InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS,
                    std::to_string(perfHintsConfig.ovPerfHintNumRequests)});
    OPENVINO_SUPPRESS_DEPRECATED_END
}

}  // namespace intel_cpu
}   // namespace ov
