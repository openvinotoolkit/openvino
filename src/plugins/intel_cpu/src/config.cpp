// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.h"

#include "cpu/cpu_config.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils/debug_capabilities.h"

#include <algorithm>
#include <map>
#include <string>

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
#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
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
    for (const auto& kvp : prop) {
        const auto& key = kvp.first;
        const auto& val = kvp.second;
        if (streamExecutorConfigKeys.end() !=
            std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key)) {
            streamExecutorConfig.set_property(key, val.as<std::string>());
            if (key == ov::affinity.name()) {
                const auto affinity_val = ov::util::from_string(val.as<std::string>(), ov::affinity);
                if (affinity_val == ov::Affinity::CORE || affinity_val == ov::Affinity::HYBRID_AWARE) {
                    enableCpuPinning = true;
                    changedCpuPinning = true;
                } else if (affinity_val == ov::Affinity::NUMA) {
                    enableCpuPinning = false;
                    changedCpuPinning = true;
                }
            }
        } else if (key == ov::hint::performance_mode.name()) {
            hintPerfMode = ov::util::from_string(val.as<std::string>(), ov::hint::performance_mode);
        } else if (key == ov::hint::num_requests.name()) {
            int val_i = -1;
            try {
                val_i = std::stoi(val.as<std::string>().c_str());
                if (val_i < 0) {
                    throw std::logic_error("wrong val");
                }
                hintNumRequests = static_cast<uint32_t>(val_i);
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               "for property key ",
                               ov::hint::num_requests.name(),
                               ". Expected only >0.");
            }
        } else if (key == ov::hint::enable_cpu_pinning.name()) {
            try {
                if (val.as<bool>()) {
                    enableCpuPinning = true;
                    changedCpuPinning = true;
                } else {
                    enableCpuPinning = false;
                    changedCpuPinning = true;
                }
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               "for property key ",
                               ov::hint::enable_cpu_pinning.name(),
                               ". Expected only true/false.");
            }
        } else if (key == ov::hint::scheduling_core_type.name()) {
            const auto core_type = ov::util::from_string(val.as<std::string>(), ov::hint::scheduling_core_type);
            if (core_type == ov::hint::SchedulingCoreType::ANY_CORE ||
                core_type == ov::hint::SchedulingCoreType::PCORE_ONLY ||
                core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) {
                schedulingCoreType = core_type;
            } else {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
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
            try {
                if (val.as<bool>()) {
                    enableHyperThreading = true;
                    changedHyperThreading = true;
                } else {
                    enableHyperThreading = false;
                    changedHyperThreading = true;
                }
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               "for property key ",
                               ov::hint::enable_hyper_threading.name(),
                               ". Expected only true/false.");
            }
        } else if (key == ov::intel_cpu::sparse_weights_decompression_rate.name()) {
            float val_f = 0.0f;
            try {
                val_f = std::stof(val.as<std::string>());
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
            try {
                collectPerfCounters = val.as<bool>();
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::enable_profiling.name(),
                               ". Expected only YES/NO");
            }
        } else if (key == ov::internal::exclusive_async_requests.name()) {
            try {
                exclusiveAsyncRequests = val.as<bool>();
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::internal::exclusive_async_requests.name(),
                               ". Expected only YES/NO");
            }
            OPENVINO_SUPPRESS_DEPRECATED_START
        } else if (key.compare(InferenceEngine::PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT) == 0) {
            // empty string means that dumping is switched off
            dumpToDot = val.as<std::string>();
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (key.compare(ov::internal::lp_transforms_mode.name()) == 0) {
            try {
                if (val.as<bool>()) {
                    lpTransformsMode = LPTransformsMode::On;
                } else {
                    lpTransformsMode = LPTransformsMode::Off;
                }
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::internal::lp_transforms_mode.name());
            }
        } else if (key == ov::device::id.name()) {
            device_id = val.as<std::string>();
            if (!device_id.empty()) {
                OPENVINO_THROW("CPU plugin supports only '' as device id");
            }
        } else if (key == ov::enforce_bf16.name()) {
            bool enable;
            try {
                enable = val.as<bool>();
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::enforce_bf16.name(),
                               ". Expected only YES/NO");
            }
            if (enable) {
                if (mayiuse(avx512_core)) {
                    inferencePrecision = ov::element::bf16;
                } else {
                    OPENVINO_THROW("Platform doesn't support BF16 format");
                }
            } else {
                inferencePrecision = ov::element::f32;
            }
            inferencePrecisionSetExplicitly = true;
        } else if (key == ov::hint::inference_precision.name()) {
            auto prec = val.as<std::string>();
            if (prec == "bf16") {
                if (mayiuse(avx512_core)) {
                    inferencePrecision = ov::element::bf16;
                    inferencePrecisionSetExplicitly = true;
                }
            } else if (prec == "f16") {
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
            } else if (prec == "f32") {
                inferencePrecision = ov::element::f32;
                inferencePrecisionSetExplicitly = true;
            } else {
                OPENVINO_THROW("Wrong value ",
                               prec,
                               " for property key ",
                               ov::hint::inference_precision.name(),
                               ". Supported values: bf16, f16, f32");
            }
        } else if (ov::intel_cpu::cpu_runtime_cache_capacity.name() == key) {
            int val_i = -1;
            try {
                val_i = std::stoi(val.as<std::string>());
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::intel_cpu::cpu_runtime_cache_capacity.name(),
                               ". Expected only integer numbers");
            }
            // any negative value will be treated
            // as zero that means disabling the cache
            rtCacheCapacity = std::max(val_i, 0);
        } else if (ov::intel_cpu::denormals_optimization.name() == key) {
            try {
                if (val.as<bool>()) {
                    denormalsOptMode = DenormalsOptMode::DO_On;
                } else {
                    denormalsOptMode = DenormalsOptMode::DO_Off;
                }
            } catch (ov::Exception&) {
                denormalsOptMode = DenormalsOptMode::DO_Keep;
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::intel_cpu::denormals_optimization.name(),
                               ". Expected only YES/NO");
            }
        } else if (key == ov::snippets_mode.name()) {
            auto mode = val.as<std::string>();
            if (mode == ov::util::to_string(ov::SnippetsMode::ENABLE))
                snippetsMode = SnippetsMode::Enable;
            else if (mode == ov::util::to_string(ov::SnippetsMode::IGNORE_CALLBACK))
                snippetsMode = SnippetsMode::IgnoreCallback;
            else if (mode == ov::util::to_string(ov::SnippetsMode::DISABLE))
                snippetsMode = SnippetsMode::Disable;
            else
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::snippets_mode.name(),
                               ". Expected values: ENABLE/DISABLE/IGNORE_CALLBACK");
        } else if (key == ov::hint::execution_mode.name()) {
            auto mode = val.as<std::string>();
            if (mode == "PERFORMANCE") {
                executionMode = ov::hint::ExecutionMode::PERFORMANCE;
            } else if (mode == "ACCURACY") {
                executionMode = ov::hint::ExecutionMode::ACCURACY;
            } else {
                OPENVINO_THROW("Wrong value ",
                               mode,
                               "for property key ",
                               ov::hint::execution_mode.name(),
                               ". Supported values: PERFORMANCE, ACCURACY");
            }
        } else {
            IE_THROW(NotFound) << "Unsupported property " << key << " by CPU plugin";
        }
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
        _config.insert({ov::internal::cpu_bind_thread.name(), "NO"});
        break;
    case IStreamsExecutor::ThreadBindingType::CORES:
        _config.insert({ov::internal::cpu_bind_thread.name(), "YES"});
        break;
    case IStreamsExecutor::ThreadBindingType::NUMA:
        _config.insert({ov::internal::cpu_bind_thread.name(), ov::util::to_string(ov::Affinity::NUMA)});
        break;
    case IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
        _config.insert({ov::internal::cpu_bind_thread.name(), ov::util::to_string(ov::Affinity::HYBRID_AWARE)});
        break;
    }
    if (collectPerfCounters == true)
        _config.insert({ov::enable_profiling.name(), "YES"});
    else
        _config.insert({ov::enable_profiling.name(), "NO"});
    if (exclusiveAsyncRequests == true)
        _config.insert({ov::internal::exclusive_async_requests.name(), "YES"});
    else
        _config.insert({ov::internal::exclusive_async_requests.name(), "NO"});

    _config.insert({ov::device::id.name(), device_id});

    _config.insert({ov::num_streams.name(), std::to_string(streamExecutorConfig._streams)});
    _config.insert({ov::inference_num_threads.name(), std::to_string(streamExecutorConfig._threads)});
    _config.insert({ov::hint::performance_mode.name(), ov::util::to_string(hintPerfMode)});
    _config.insert({ov::hint::num_requests.name(), std::to_string(hintNumRequests)});

    if (inferencePrecision == ov::element::bf16) {
        _config.insert({ov::enforce_bf16.name(), "YES"});
    } else {
        _config.insert({ov::enforce_bf16.name(), "NO"});
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    _config.insert({InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                    std::to_string(streamExecutorConfig._streams)});
    _config.insert(
        {InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, std::to_string(streamExecutorConfig._threads)});
    _config.insert({InferenceEngine::PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT, dumpToDot});
    OPENVINO_SUPPRESS_DEPRECATED_END
}

}  // namespace intel_cpu
}   // namespace ov
