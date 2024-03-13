// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.h"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils/debug_capabilities.h"
#include "utils/precision_support.h"
#include "utils/cpu_utils.hpp"

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
    threadBindingType = IStreamsExecutor::NONE;
#else
    threadBindingType = IStreamsExecutor::CORES;
#endif

// for the TBB code-path, additional configuration depending on the OS and CPU types
#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
#    if defined(__APPLE__) || defined(_WIN32)
    // 'CORES' is not implemented for Win/MacOS; so the 'NONE' or 'NUMA' is default
    auto numaNodes = get_available_numa_nodes();
    if (numaNodes.size() > 1) {
        threadBindingType = IStreamsExecutor::NUMA;
    } else {
        threadBindingType = IStreamsExecutor::NONE;
    }
#    endif

    if (get_available_cores_types().size() > 1 /*Hybrid CPU*/) {
        threadBindingType = IStreamsExecutor::HYBRID_AWARE;
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
            streams = streamExecutorConfig.get_streams();
            threads = streamExecutorConfig.get_threads();
            threadsPerStream = streamExecutorConfig.get_threads_per_stream();
            if (key == ov::num_streams.name()) {
                ov::Any value = val.as<std::string>();
                auto streams_value = value.as<ov::streams::Num>();
                if (streams_value == ov::streams::NUMA) {
                    latencyThreadingMode = Config::LatencyThreadingMode::PER_NUMA_NODE;
                } else if (streams_value == ov::streams::AUTO) {
                    hintPerfMode = ov::hint::PerformanceMode::THROUGHPUT;
                    changedHintPerfMode = true;
                } else {
                    streamsChanged = true;
                }
            }
            OPENVINO_SUPPRESS_DEPRECATED_START
        } else if (key == ov::affinity.name()) {
            try {
                changedCpuPinning = true;
                ov::Affinity affinity = val.as<ov::Affinity>();
#if defined(__APPLE__)
                enableCpuPinning = false;
                threadBindingType = affinity == ov::Affinity::NONE ? IStreamsExecutor::ThreadBindingType::NONE
                                                                   : IStreamsExecutor::ThreadBindingType::NUMA;
#else
                enableCpuPinning =
                    (affinity == ov::Affinity::CORE || affinity == ov::Affinity::HYBRID_AWARE) ? true : false;
                switch (affinity) {
                case ov::Affinity::NONE:
                    threadBindingType = IStreamsExecutor::ThreadBindingType::NONE;
                    break;
                case ov::Affinity::CORE: {
                    threadBindingType = IStreamsExecutor::ThreadBindingType::CORES;
                } break;
                case ov::Affinity::NUMA:
                    threadBindingType = IStreamsExecutor::ThreadBindingType::NUMA;
                    break;
                case ov::Affinity::HYBRID_AWARE:
                    threadBindingType = IStreamsExecutor::ThreadBindingType::HYBRID_AWARE;
                    break;
                default:
                    OPENVINO_THROW("Wrong value ",
                                   val.as<std::string>(),
                                   "for property key ",
                                   key,
                                   ". Expected only ov::Affinity::CORE/NUMA/HYBRID_AWARE.");
                }
#endif
            } catch (const ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               "for property key ",
                               key,
                               ". Expected only ov::Affinity::CORE/NUMA/HYBRID_AWARE.");
            }
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (key == ov::hint::performance_mode.name()) {
            try {
                hintPerfMode = !changedHintPerfMode ? val.as<ov::hint::PerformanceMode>() : hintPerfMode;
            } catch (const ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               "for property key ",
                               key,
                               ". Expected only ov::hint::PerformanceMode::LATENCY/THROUGHPUT/CUMULATIVE_THROUGHPUT.");
            }
        } else if (key == ov::log::level.name()) {
            try {
                logLevel = val.as<ov::log::Level>();
            } catch (const ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                        val.as<std::string>(),
                        " for property key ",
                        key,
                        ". Expected only ov::log::Level::NO/ERR/WARNING/INFO/DEBUG/TRACE.");
            }
        } else if (key == ov::hint::num_requests.name()) {
            try {
                ov::Any value = val.as<std::string>();
                int val_i = value.as<int>();
                if (val_i < 0)
                    OPENVINO_THROW("invalid value.");
                hintNumRequests = static_cast<uint32_t>(val_i);
            } catch (const ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               "for property key ",
                               ov::hint::num_requests.name(),
                               ". Expected only > 0.");
            }
        } else if (key == ov::hint::enable_cpu_pinning.name()) {
            try {
                enableCpuPinning = val.as<bool>();
                changedCpuPinning = true;
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               "for property key ",
                               ov::hint::enable_cpu_pinning.name(),
                               ". Expected only true/false.");
            }
        } else if (key == ov::hint::scheduling_core_type.name()) {
            try {
                schedulingCoreType = val.as<ov::hint::SchedulingCoreType>();
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               "for property key ",
                               ov::hint::scheduling_core_type.name(),
                               ". Expected only ",
                               ov::hint::SchedulingCoreType::ANY_CORE,
                               '/',
                               ov::hint::SchedulingCoreType::PCORE_ONLY,
                               '/',
                               ov::hint::SchedulingCoreType::ECORE_ONLY);
            }
        } else if (key == ov::hint::enable_hyper_threading.name()) {
            try {
                enableHyperThreading = val.as<bool>();
                changedHyperThreading = true;
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
                val_f = val.as<float>();
            } catch (const ov::Exception&) {
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
        } else if (key == ov::hint::dynamic_quantization_group_size.name()) {
            try {
                fcDynamicQuantizationGroupSize = val.as<uint64_t>();
            } catch (const ov::Exception&) {
                OPENVINO_THROW("Wrong value for property key ",
                                ov::hint::dynamic_quantization_group_size.name(),
                                ". Expected only unsinged integer numbers");
            }
        } else if (key == ov::enable_profiling.name()) {
            try {
                collectPerfCounters = val.as<bool>();
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::enable_profiling.name(),
                               ". Expected only true/false");
            }
        } else if (key == ov::internal::exclusive_async_requests.name()) {
            try {
                exclusiveAsyncRequests = val.as<bool>();
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::internal::exclusive_async_requests.name(),
                               ". Expected only true/false");
            }
        } else if (key == ov::intel_cpu::lp_transforms_mode.name()) {
            try {
                lpTransformsMode = val.as<bool>() ? LPTransformsMode::On : LPTransformsMode::Off;
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               key,
                               ". Expected value only ov::intel_cpu::Config::LPTransformsMode::On/Off");
            }
        } else if (key == ov::device::id.name()) {
            device_id = val.as<std::string>();
            if (!device_id.empty()) {
                OPENVINO_THROW("CPU plugin supports only '' as device id");
            }
        } else if (key == ov::hint::inference_precision.name()) {
            try {
                auto const prec = val.as<ov::element::Type>();
                inferencePrecisionSetExplicitly = true;
                if (prec == ov::element::bf16) {
                    if (hasHardwareSupport(ov::element::bf16)) {
                        inferencePrecision = ov::element::bf16;
                    }
                } else if (prec == ov::element::f16) {
#if defined(OPENVINO_ARCH_X86_64)
                    if (hasHardwareSupport(ov::element::f16)) {
                        inferencePrecision = ov::element::f16;
                    }
#elif defined(OV_CPU_ARM_ENABLE_FP16)
                    // TODO: add runtime FP16 feature support check for ARM
                    inferencePrecision = ov::element::f16;
#endif
                } else if (prec == ov::element::f32) {
                    inferencePrecision = ov::element::f32;
                } else {
                    OPENVINO_THROW("invalid value");
                }
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::hint::inference_precision.name(),
                               ". Supported values: bf16, f16, f32");
            }
        } else if (ov::intel_cpu::cpu_runtime_cache_capacity.name() == key) {
            int val_i = -1;
            try {
                ov::Any value = val.as<std::string>();
                val_i = value.as<int>();
            } catch (const ov::Exception&) {
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
                denormalsOptMode = val.as<bool>() ? DenormalsOptMode::DO_On : DenormalsOptMode::DO_Off;
            } catch (ov::Exception&) {
                denormalsOptMode = DenormalsOptMode::DO_Keep;
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::intel_cpu::denormals_optimization.name(),
                               ". Expected only true/false");
            }
        } else if (key == ov::intel_cpu::snippets_mode.name()) {
            try {
                auto const mode = val.as<ov::intel_cpu::SnippetsMode>();
                if (mode == ov::intel_cpu::SnippetsMode::ENABLE)
                    snippetsMode = SnippetsMode::Enable;
                else if (mode == ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK)
                    snippetsMode = SnippetsMode::IgnoreCallback;
                else if (mode == ov::intel_cpu::SnippetsMode::DISABLE)
                    snippetsMode = SnippetsMode::Disable;
                else
                    OPENVINO_THROW("invalid value");
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::intel_cpu::snippets_mode.name(),
                               ". Expected values: ov::intel_cpu::SnippetsMode::ENABLE/DISABLE/IGNORE_CALLBACK");
            }
        } else if (key == ov::hint::execution_mode.name()) {
            try {
                executionMode = val.as<ov::hint::ExecutionMode>();
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               "for property key ",
                               ov::hint::execution_mode.name(),
                               ". Supported values: ov::hint::ExecutionMode::PERFORMANCE/ACCURACY");
            }
        } else if (key == ov::hint::kv_cache_precision.name()) {
            try {
                auto const prec = val.as<ov::element::Type>();
                if (one_of(prec, ov::element::f32, ov::element::f16, ov::element::bf16, ov::element::u8)) {
                    kvCachePrecision = prec;
                } else {
                     OPENVINO_THROW("invalid value");
                }
            } catch (ov::Exception&) {
                OPENVINO_THROW("Wrong value ",
                               val.as<std::string>(),
                               " for property key ",
                               ov::hint::kv_cache_precision.name(),
                               ". Supported values: u8, bf16, f16, f32");
            }
        } else {
            OPENVINO_THROW("NotFound: Unsupported property ", key, " by CPU plugin.");
        }
    }
    // apply execution mode after all the params are handled to prevent possible conflicts
    // when both execution_mode and inference_precision are specified
    if (!inferencePrecisionSetExplicitly) {
        if (executionMode == ov::hint::ExecutionMode::PERFORMANCE) {
            inferencePrecision = ov::element::f32;
#if defined(OV_CPU_ARM_ENABLE_FP16)
            // fp16 precision is used as default precision on ARM for non-convolution networks
            // fp16 ACL convolution is slower than fp32
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
        streams = 1;
        streamsChanged = true;
    }

    this->modelType = modelType;

    CPU_DEBUG_CAP_ENABLE(applyDebugCapsProperties());
    updateProperties();
}

void Config::updateProperties() {
    if (!_config.empty())
        return;

    if (collectPerfCounters == true)
        _config.insert({ov::enable_profiling.name(), "YES"});
    else
        _config.insert({ov::enable_profiling.name(), "NO"});
    if (exclusiveAsyncRequests == true)
        _config.insert({ov::internal::exclusive_async_requests.name(), "YES"});
    else
        _config.insert({ov::internal::exclusive_async_requests.name(), "NO"});

    _config.insert({ov::device::id.name(), device_id});

    _config.insert({ov::hint::performance_mode.name(), ov::util::to_string(hintPerfMode)});
    _config.insert({ov::hint::num_requests.name(), std::to_string(hintNumRequests)});
}

}  // namespace intel_cpu
}  // namespace ov
