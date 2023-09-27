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
#include "openvino/runtime/properties.hpp"
#include "utils/debug_capabilities.h"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov {
namespace intel_cpu {

using namespace InferenceEngine;
using namespace dnnl::impl::cpu::x64;

Config::Config() {
    // this is default mode
#if defined(__APPLE__) || defined(_WIN32)
    streamExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::NONE;
#else
    streamExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::CORES;
#endif

// for the TBB code-path, additional configuration depending on the OS and CPU types
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
#    if defined(__APPLE__) || defined(_WIN32)
    // 'CORES' is not implemented for Win/MacOS; so the 'NONE' or 'NUMA' is default
    auto numaNodes = getAvailableNUMANodes();
    if (numaNodes.size() > 1) {
        streamExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::NUMA;
    } else {
        streamExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::NONE;
    }
#    endif

    if (getAvailableCoresTypes().size() > 1 /*Hybrid CPU*/) {
        streamExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::HYBRID_AWARE;
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

void Config::readProperties(const std::map<std::string, std::string> &prop, const ModelType modelType) {
    const auto streamExecutorConfigKeys = streamExecutorConfig.SupportedKeys();
    const auto hintsConfigKeys = perfHintsConfig.SupportedKeys();
    for (const auto& kvp : prop) {
        const auto& key = kvp.first;
        const auto& val = kvp.second;
        IE_SUPPRESS_DEPRECATED_START
        if (streamExecutorConfigKeys.end() !=
            std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key)) {
            streamExecutorConfig.SetConfig(key, val);
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
            if (val == PluginConfigParams::YES) {
                enableCpuPinning = true;
                changedCpuPinning = true;
            } else if (val == PluginConfigParams::NO) {
                enableCpuPinning = false;
                changedCpuPinning = true;
            } else {
                IE_THROW() << "Wrong value " << val << "for property key " << ov::hint::enable_cpu_pinning.name()
                           << ". Expected only true/false." << std::endl;
            }
        } else if (key == ov::hint::scheduling_core_type.name()) {
            const auto core_type = ov::util::from_string(val, ov::hint::scheduling_core_type);
            if (core_type == ov::hint::SchedulingCoreType::ANY_CORE ||
                core_type == ov::hint::SchedulingCoreType::PCORE_ONLY ||
                core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) {
                schedulingCoreType = core_type;
            } else {
                IE_THROW() << "Wrong value " << val << "for property key " << ov::hint::scheduling_core_type.name()
                           << ". Expected only " << ov::hint::SchedulingCoreType::ANY_CORE << "/"
                           << ov::hint::SchedulingCoreType::PCORE_ONLY << "/"
                           << ov::hint::SchedulingCoreType::ECORE_ONLY << std::endl;
            }
        } else if (key == ov::hint::enable_hyper_threading.name()) {
            if (val == PluginConfigParams::YES) {
                enableHyperThreading = true;
                changedHyperThreading = true;
            } else if (val == PluginConfigParams::NO) {
                enableHyperThreading = false;
                changedHyperThreading = true;
            } else {
                IE_THROW() << "Wrong value " << val << "for property key " << ov::hint::enable_hyper_threading.name()
                           << ". Expected only true/false." << std::endl;
            }
        } else if (key == CPUConfigParams::KEY_CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE) {
            float val_f = 0.0f;
            try {
                val_f = std::stof(val);
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << CPUConfigParams::KEY_CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE
                                    << ". Expected only float numbers";
            }
            if (val_f < 0.f || val_f > 1.f) {
                IE_THROW() << "Wrong value for property key " << CPUConfigParams::KEY_CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE
                                    << ". Sparse rate must be in range [0.0f,1.0f]";
            } else {
                fcSparseWeiDecompressionRate = val_f;
            }
        } else if (key == PluginConfigParams::KEY_PERF_COUNT) {
            if (val == PluginConfigParams::YES) collectPerfCounters = true;
            else if (val == PluginConfigParams::NO) collectPerfCounters = false;
            else
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_PERF_COUNT
                                   << ". Expected only YES/NO";
        } else if (key == PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) {
            if (val == PluginConfigParams::YES) exclusiveAsyncRequests = true;
            else if (val == PluginConfigParams::NO) exclusiveAsyncRequests = false;
            else
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS
                                   << ". Expected only YES/NO";
            IE_SUPPRESS_DEPRECATED_START
        } else if (key.compare(PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT) == 0) {
            IE_SUPPRESS_DEPRECATED_END
            // empty string means that dumping is switched off
            dumpToDot = val;
        } else if (key.compare(PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE) == 0) {
            if (val == PluginConfigParams::NO)
                lpTransformsMode = LPTransformsMode::Off;
            else if (val == PluginConfigParams::YES)
                lpTransformsMode = LPTransformsMode::On;
            else
                IE_THROW() << "Wrong value for property key " << PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE;
        } else if (key == ov::device::id.name()) {
            device_id = val;
            if (!device_id.empty()) {
                IE_THROW() << "CPU plugin supports only '' as device id";
            }
        } else if (key == PluginConfigParams::KEY_ENFORCE_BF16) {
            if (val == PluginConfigParams::YES) {
                if (mayiuse(avx512_core)) {
                    inferencePrecision = ov::element::bf16;
                } else {
                    IE_THROW() << "Platform doesn't support BF16 format";
                }
            } else if (val == PluginConfigParams::NO) {
                inferencePrecision = ov::element::f32;
            } else {
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_ENFORCE_BF16
                    << ". Expected only YES/NO";
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
                IE_THROW() << "Wrong value for property key " << ov::hint::inference_precision.name()
                    << ". Supported values: bf16, f16, f32";
            }
        } else if (PluginConfigInternalParams::KEY_CPU_RUNTIME_CACHE_CAPACITY == key) {
            int val_i = -1;
            try {
                val_i = std::stoi(val);
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << PluginConfigInternalParams::KEY_CPU_RUNTIME_CACHE_CAPACITY
                           << ". Expected only integer numbers";
            }
            // any negative value will be treated
            // as zero that means disabling the cache
            rtCacheCapacity = std::max(val_i, 0);
        } else if (CPUConfigParams::KEY_CPU_DENORMALS_OPTIMIZATION == key) {
            if (val == PluginConfigParams::YES) {
                denormalsOptMode = DenormalsOptMode::DO_On;
                changedDenormalsOptMode = true;
            } else if (val == PluginConfigParams::NO) {
                denormalsOptMode = DenormalsOptMode::DO_Off;
                changedDenormalsOptMode = true;
            } else {
                denormalsOptMode = DenormalsOptMode::DO_Keep;
                IE_THROW() << "Wrong value for property key " << CPUConfigParams::KEY_CPU_DENORMALS_OPTIMIZATION
                << ". Expected only YES/NO";
            }
            streamExecutorConfig.optDenormalsForTBB = !changedDenormalsOptMode;
        } else if (key == PluginConfigInternalParams::KEY_SNIPPETS_MODE) {
            if (val == PluginConfigInternalParams::ENABLE)
                snippetsMode = SnippetsMode::Enable;
            else if (val == PluginConfigInternalParams::IGNORE_CALLBACK)
                snippetsMode = SnippetsMode::IgnoreCallback;
            else if (val == PluginConfigInternalParams::DISABLE)
                snippetsMode = SnippetsMode::Disable;
            else
                IE_THROW() << "Wrong value for property key " << PluginConfigInternalParams::KEY_SNIPPETS_MODE
                            << ". Expected values: ENABLE/DISABLE/IGNORE_CALLBACK";
        } else if (key == ov::hint::execution_mode.name()) {
            if (val == "PERFORMANCE") {
                executionMode = ov::hint::ExecutionMode::PERFORMANCE;
            } else if (val == "ACCURACY") {
                executionMode = ov::hint::ExecutionMode::ACCURACY;
            } else {
                IE_THROW() << "Wrong value for property key " << ov::hint::execution_mode.name()
                    << ". Supported values: PERFORMANCE, ACCURACY";
            }
        } else {
            IE_THROW(NotFound) << "Unsupported property " << key << " by CPU plugin";
        }
        IE_SUPPRESS_DEPRECATED_END
    }
    // apply execution mode after all the params are handled to prevent possible conflicts
    // when both execution_mode and inference_precision are specified
    if (!inferencePrecisionSetExplicitly) {
        if (executionMode == ov::hint::ExecutionMode::PERFORMANCE) {
            inferencePrecision = ov::element::f32;
#if defined(OV_CPU_ARM_ENABLE_FP16)
            //fp16 precision is used as default precision on ARM for non-convolution networks
            //fp16 ACL convolution is slower than fp32
            if (modelType != ModelType::CNN)
                inferencePrecision = ov::element::f16;
#else
            if (mayiuse(avx512_core_bf16))
                inferencePrecision = ov::element::bf16;
#endif
        } else {
            inferencePrecision = ov::element::f32;
        }
    }

    if (!prop.empty())
        _config.clear();

    if (exclusiveAsyncRequests) { // Exclusive request feature disables the streams
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
        _config.insert({ PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::NO });
        break;
    case IStreamsExecutor::ThreadBindingType::CORES:
        _config.insert({ PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::YES });
        break;
    case IStreamsExecutor::ThreadBindingType::NUMA:
        _config.insert({ PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::NUMA });
        break;
    case IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
        _config.insert({ PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::HYBRID_AWARE });
        break;
    }
    if (collectPerfCounters == true)
        _config.insert({ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES });
    else
        _config.insert({ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::NO });
    if (exclusiveAsyncRequests == true)
        _config.insert({ PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, PluginConfigParams::YES });
    else
        _config.insert({ PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, PluginConfigParams::NO });

    _config.insert({ PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, std::to_string(streamExecutorConfig._streams) });

    _config.insert({ PluginConfigParams::KEY_CPU_THREADS_NUM, std::to_string(streamExecutorConfig._threads) });

    _config.insert({ PluginConfigParams::KEY_DEVICE_ID, device_id });

    IE_SUPPRESS_DEPRECATED_START
        _config.insert({ PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT, dumpToDot });
    IE_SUPPRESS_DEPRECATED_END;
    if (inferencePrecision == ov::element::bf16) {
        _config.insert({ PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES });
    } else {
        _config.insert({ PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO });
    }

    _config.insert({ PluginConfigParams::KEY_PERFORMANCE_HINT, perfHintsConfig.ovPerfHint });
    _config.insert({ PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS,
            std::to_string(perfHintsConfig.ovPerfHintNumRequests) });
}

}  // namespace intel_cpu
}   // namespace ov
