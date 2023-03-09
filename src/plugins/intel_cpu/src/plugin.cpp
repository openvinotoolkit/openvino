// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp" // must be included first

#include "plugin.h"

#include "transformation_pipeline.h"
#include "itt.h"
#include "extension_mngr.h"
#include "extension.h"
#include "serialize.h"
#include "threading/ie_executor_manager.hpp"

#include "ie_icore.hpp"
#include "ie_plugin_config.hpp"
#include "ie_system_conf.h"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

#include <ie_ngraph_utils.hpp>

#include "performance_heuristics.hpp"

#include "weights_cache.hpp"
#include "utils/denormals.hpp"

#if defined(__linux__)
# include <sys/auxv.h>
# include <signal.h>
# include <sys/mman.h>
#endif

#include <cpu/x64/cpu_isa_traits.hpp>
#include <itt.h>

using namespace InferenceEngine;

#define IE_CPU_PLUGIN_THROW(...) IE_THROW(__VA_ARGS__) << "CPU plugin: "

namespace ov {
namespace intel_cpu {

static std::string getDeviceFullName() {
    std::string brand_string;
#if defined(__EMSCRIPTEN___)
    brand_string = "WebAssembly CPU";
#elif defined(OPENVINO_ARCH_RISCV64)
    // TODO: extract actual device name
    brand_string = "RISCV-64 CPU";
#elif defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    // TODO: extract actual device name
    brand_string = "ARM CPU";
#elif defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    const unsigned int addr_list[3] = { 0x80000002, 0x80000003, 0x80000004 };
    unsigned int regs[4];
    for (auto addr : addr_list) {
        regs[0] = addr;
#ifdef _WIN32
        __cpuid(reinterpret_cast<int*>(regs), regs[0]);
#else
        __cpuid(regs[0], regs[0], regs[1], regs[2], regs[3]);
#endif
        char *ch = reinterpret_cast<char*>(&regs[0]);
        for (size_t j = 0; j < sizeof(regs); j++)
            brand_string += ch[j];
    }
#else
# error "Unkown CPU architecture. Please, add support to openvino/core/visibility.hpp"
#endif
    return brand_string;
}

#if defined(__linux__)

#ifndef AT_MINSIGSTKSZ
# define AT_MINSIGSTKSZ 51
#endif

class SigAltStackSetup {
    stack_t new_stack{0};
    stack_t old_stack{0};

public:
    SigAltStackSetup() {
        memset(&old_stack, 0, sizeof(old_stack));
        memset(&new_stack, 0, sizeof(new_stack));

        auto minsigstksz = getauxval(AT_MINSIGSTKSZ);
        auto new_size = minsigstksz + SIGSTKSZ;
        void * altstack =  mmap(NULL, new_size, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
        if (altstack == MAP_FAILED) {
            return;
        }
        new_stack.ss_size = new_size;
        new_stack.ss_sp = altstack;
        auto rc = sigaltstack(&new_stack, &old_stack);
        if (rc) {
            munmap(new_stack.ss_sp, new_stack.ss_size);
            new_stack.ss_sp = nullptr;
            new_stack.ss_size = 0;
            return;
        }
    }

    ~SigAltStackSetup() {
        stack_t current_stack;
        if (new_stack.ss_sp) {
            // restore old stack if new_stack is still the current one
            if (sigaltstack(NULL, &current_stack) == 0) {
                if (current_stack.ss_sp == new_stack.ss_sp) {
                    sigaltstack(&old_stack, NULL);
                }
            }
            munmap(new_stack.ss_sp, new_stack.ss_size);
            new_stack.ss_sp = nullptr;
            new_stack.ss_size = 0;
        }
    }
};

class CPUSpecialSetup {
    SigAltStackSetup ss;

public:
    CPUSpecialSetup() = default;
};
#else // __linux__
class CPUSpecialSetup {
public:
    CPUSpecialSetup() = default;
};
#endif // __linux__

Engine::Engine() :
    deviceFullName(getDeviceFullName()),
    specialSetup(new CPUSpecialSetup) {
    _pluginName = "CPU";
    extensionManager->AddExtension(std::make_shared<Extension>());
}

Engine::~Engine() {
    executorManager()->clear("CPU");
    executorManager()->clear("CPUStreamsExecutor");
    executorManager()->clear("CPUCallbackExecutor");
}

static bool streamsSet(const std::map<std::string, std::string>& config) {
    return config.count(PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS) ||
           config.count(ov::num_streams.name());
}

void Engine::ApplyPerformanceHints(std::map<std::string, std::string> &config, const std::shared_ptr<ngraph::Function>& ngraphFunc) const {
    auto getNumStreamsLatency = [&]() {
        return std::pair<std::string, std::string>(CONFIG_VALUE(CPU_THROUGHPUT_NUMA), ov::util::to_string(ov::streams::NUMA));
    };

    auto getNumStreamsThroughput = [&]() {
        const auto isa = dnnl::get_effective_cpu_isa();
        float isaSpecificThreshold = 1.0f;
        switch (isa) {
        case dnnl::cpu_isa::sse41 :
            isaSpecificThreshold = 0.5f;
            break;
        case dnnl::cpu_isa::avx2:
        case dnnl::cpu_isa::avx512_core:
            isaSpecificThreshold = 1.0f;
            break;
        case dnnl::cpu_isa::avx512_core_vnni:
        case dnnl::cpu_isa::avx2_vnni:
            isaSpecificThreshold = 2.0f;
            break;
        case dnnl::cpu_isa::avx512_core_amx:
            isaSpecificThreshold = 4.0f;
            break;
        default:
            isaSpecificThreshold = 1.0f;
        }
        // the more "capable" the CPU in general, the more streams we may want to keep to keep it utilized
        const float memThresholdAssumeLimitedForISA = ov::MemBandwidthPressure::LIMITED/isaSpecificThreshold;
        const float L2_cache_size = dnnl::utils::get_cache_size(2 /*level*/, true /*per core */);
        ov::MemBandwidthPressure networkToleranceForLowCache = ov::MemBandwidthPressureTolerance(
            ngraphFunc,
            L2_cache_size, memThresholdAssumeLimitedForISA);
        const auto default_streams = GetNumStreams(engConfig.streamExecutorConfig._threadBindingType,
                                                   IStreamsExecutor::Config::StreamMode::DEFAULT,
                                                   engConfig.streamExecutorConfig._enable_hyper_thread);
        auto streams_info = default_streams;
        if (networkToleranceForLowCache.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
            if ((networkToleranceForLowCache.ratio_compute_convs == ov::MemBandwidthPressure::ALL)
                || (networkToleranceForLowCache.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL)) {
                // all relevant layers (convs, etc) are compute-limited, the most aggressive val for #streams
                streams_info = GetNumStreams(engConfig.streamExecutorConfig._threadBindingType,
                                             IStreamsExecutor::Config::StreamMode::AGGRESSIVE,
                                             engConfig.streamExecutorConfig._enable_hyper_thread);
            }   // otherwise (no recognized layers) falling back to the default value
        } else if (networkToleranceForLowCache.max_mem_tolerance > memThresholdAssumeLimitedForISA) {
            // network is below the ISA-specific threshold
            streams_info = GetNumStreams(engConfig.streamExecutorConfig._threadBindingType,
                                         IStreamsExecutor::Config::StreamMode::AGGRESSIVE,
                                         engConfig.streamExecutorConfig._enable_hyper_thread);
        } else if (networkToleranceForLowCache.max_mem_tolerance > ov::MemBandwidthPressure::LIMITED) {
            // network is below general threshold
            streams_info = GetNumStreams(engConfig.streamExecutorConfig._threadBindingType,
                                         IStreamsExecutor::Config::StreamMode::LESSAGGRESSIVE,
                                         engConfig.streamExecutorConfig._enable_hyper_thread);
            streams_info.num_streams = std::max(default_streams.num_streams, streams_info.num_streams);
        }
        auto num_requests = config.find(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS));
        if (num_requests != config.end()) {  // arrived with config to the LoadNetwork (and thus higher pri)
            auto val = PerfHintsConfig::CheckPerformanceHintRequestValue(num_requests->second);
            if (val > 0)
                streams_info.num_streams = std::min(streams_info.num_streams, val);
        } else if (engConfig.perfHintsConfig.ovPerfHintNumRequests) {  // set thru SetConfig to the plugin, 2nd priority
            streams_info.num_streams =
                std::min(streams_info.num_streams, engConfig.perfHintsConfig.ovPerfHintNumRequests);
        }
        return std::pair<std::string, Engine::StreamCfg>(std::to_string(streams_info.num_streams), streams_info);
    };

    auto getPerfHintName = [&]() {
        const bool streamsExplicitlySetForModel = streamsSet(config);
        // checking streams (to avoid overriding what user might explicitly set in the incoming config or previously via SetConfig)
        if (streamsExplicitlySetForModel ||
            streamsExplicitlySetForEngine)
            return std::string();

        const auto& perf_hint = config.find(CONFIG_KEY(PERFORMANCE_HINT));
        // the perf_hint may have just arrived to the LoadNetwork, or was set with the plugin's SetConfig
        if (perf_hint == config.end() && engConfig.perfHintsConfig.ovPerfHint.empty())
            return std::string();
        /* performance hints set for network has higher pririty than engine ones.
        * This applies for all the configuration parameters */
        const auto perf_hint_name = (perf_hint != config.end()) ?
            PerfHintsConfig::CheckPerformanceHintValue(perf_hint->second) :
            engConfig.perfHintsConfig.ovPerfHint;
        return perf_hint_name;
    };

    // We compute both hints values because the optimal number of streams are computed based on ov::Model
    // while we export model in cpu internal opset so we need to save precomputed optimal # streams for both hint modes
    const auto latency_hints = getNumStreamsLatency();
    const auto tput_hints = getNumStreamsThroughput();

    // save hints parameters to model rt_info
    ov::AnyMap hints_props;
    const auto latency_name = std::string(CONFIG_VALUE(LATENCY)) + "_" + std::string(ov::num_streams.name());
    const auto tput_name = std::string(CONFIG_VALUE(THROUGHPUT)) + "_" + std::string(ov::num_streams.name());
    hints_props.insert({latency_name, latency_hints.second});
    hints_props.insert({tput_name, std::to_string(tput_hints.second.num_streams)});
    ngraphFunc->set_rt_info(hints_props, "intel_cpu_hints_config");

    const auto perf_hint_name = getPerfHintName();
    if (perf_hint_name == CONFIG_VALUE(LATENCY)) {
        config[CONFIG_KEY(CPU_THROUGHPUT_STREAMS)] = latency_hints.first;
        config[ov::num_streams.name()] = latency_hints.second;
    } else if (perf_hint_name == CONFIG_VALUE(THROUGHPUT)) {
        config[CONFIG_KEY(CPU_THROUGHPUT_STREAMS)] = tput_hints.first;
        config[ov::num_streams.name()] = tput_hints.first;
        config[CONFIG_KEY_INTERNAL(BIG_CORE_STREAMS)] = std::to_string(tput_hints.second.big_core_streams);
        config[CONFIG_KEY_INTERNAL(SMALL_CORE_STREAMS)] = std::to_string(tput_hints.second.small_core_streams);
        config[CONFIG_KEY_INTERNAL(THREADS_PER_STREAM_BIG)] = std::to_string(tput_hints.second.threads_per_stream_big);
        config[CONFIG_KEY_INTERNAL(THREADS_PER_STREAM_SMALL)] =
            std::to_string(tput_hints.second.threads_per_stream_small);
        config[CONFIG_KEY_INTERNAL(SMALL_CORE_OFFSET)] = std::to_string(tput_hints.second.small_core_offset);
    }
}

Engine::StreamCfg Engine::GetNumStreams(InferenceEngine::IStreamsExecutor::ThreadBindingType thread_binding_type,
                                        int stream_mode,
                                        const bool enable_hyper_thread) const {
    const int sockets = static_cast<int>(getAvailableNUMANodes().size());
    const int num_cores =
        thread_binding_type == InferenceEngine::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE
            ? parallel_get_max_threads()
            : (sockets == 1 ? (enable_hyper_thread ? parallel_get_max_threads() : getNumberOfCPUCores())
                            : getNumberOfCPUCores());
    const int num_cores_phy = getNumberOfCPUCores();
    const int num_big_cores_phy = getNumberOfCPUCores(true);
    const int num_small_cores = num_cores_phy - num_big_cores_phy;
    const int num_big_cores = num_cores > num_cores_phy ? num_big_cores_phy * 2 : num_big_cores_phy;
    StreamCfg stream_cfg = {0};

    if (stream_mode == DEFAULT) {
        // bare minimum of streams (that evenly divides available number of core)
        if (thread_binding_type == InferenceEngine::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE) {
            if (0 == num_big_cores_phy % 4) {
                stream_cfg.threads_per_stream_big = 4;
            } else if (0 == num_big_cores_phy % 5) {
                stream_cfg.threads_per_stream_big = 5;
            } else if (0 == num_big_cores_phy % 3) {
                stream_cfg.threads_per_stream_big = 3;
            } else {  // if user disables some cores say in BIOS, so we got weird #cores which is not easy to divide
                stream_cfg.threads_per_stream_big = num_big_cores_phy;
            }

            stream_cfg.big_core_streams = num_big_cores / stream_cfg.threads_per_stream_big;
            stream_cfg.threads_per_stream_small = stream_cfg.threads_per_stream_big;
            if (num_small_cores == 0) {
                stream_cfg.threads_per_stream_small = 0;
            } else if (num_small_cores < stream_cfg.threads_per_stream_small) {
                stream_cfg.small_core_streams = 1;
                stream_cfg.threads_per_stream_small = num_small_cores;
                stream_cfg.threads_per_stream_big = stream_cfg.threads_per_stream_small;
                // Balance the computation of physical core and logical core, the number of threads on the physical core
                // and logical core should be equal
                stream_cfg.big_core_streams = num_big_cores_phy / stream_cfg.threads_per_stream_big * 2;
            } else {
                stream_cfg.small_core_streams = num_small_cores / stream_cfg.threads_per_stream_small;
            }
        } else {
            if (0 == num_cores % 4)
                stream_cfg.num_streams = std::max(4, num_cores / 4);
            else if (0 == num_cores % 5)
                stream_cfg.num_streams = std::max(5, num_cores / 5);
            else if (0 == num_cores % 3)
                stream_cfg.num_streams = std::max(3, num_cores / 3);
            else  // if user disables some cores say in BIOS, so we got weird #cores which is not easy to divide
                stream_cfg.num_streams = 1;
        }
    } else if (stream_mode == AGGRESSIVE) {
        if (thread_binding_type == InferenceEngine::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE) {
            stream_cfg.big_core_streams = num_big_cores;
            stream_cfg.small_core_streams = num_small_cores;
            stream_cfg.threads_per_stream_big = num_big_cores / stream_cfg.big_core_streams;
            stream_cfg.threads_per_stream_small =
                num_small_cores == 0 ? 0 : num_small_cores / stream_cfg.small_core_streams;
        } else {
            stream_cfg.num_streams = num_cores_phy;
        }
    } else if (stream_mode == LESSAGGRESSIVE) {
        if (thread_binding_type == InferenceEngine::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE) {
            stream_cfg.big_core_streams = num_big_cores / 2;
            stream_cfg.small_core_streams = num_small_cores / 2;
            stream_cfg.threads_per_stream_big = num_big_cores / stream_cfg.big_core_streams;
            stream_cfg.threads_per_stream_small =
                num_small_cores == 0 ? 0 : num_small_cores / stream_cfg.small_core_streams;
        } else {
            stream_cfg.num_streams = num_cores_phy / 2;
        }
    } else {
        IE_THROW() << "Wrong stream mode to get num of streams: " << stream_mode;
    }
    stream_cfg.num_streams = stream_cfg.num_streams > 0
                                 ? stream_cfg.num_streams
                                 : stream_cfg.big_core_streams + stream_cfg.small_core_streams;
    stream_cfg.small_core_offset = num_small_cores == 0 ? 0 : num_big_cores;
    return stream_cfg;
}

InferenceEngine::IExecutableNetworkInternal::Ptr
Engine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network, const std::map<std::string, std::string> &orig_config) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "Engine::LoadExeNetworkImpl");
    CREATE_DEBUG_TIMER(debugLoadTimer);

    // verification of supported input
    for (const auto &ii : network.getInputsInfo()) {
        auto input_precision = ii.second->getPrecision();

        using hash_t = std::hash<typename std::underlying_type<Precision::ePrecision>::type>;

        static const std::unordered_set<Precision::ePrecision, hash_t> supported_precisions = {
            Precision::U8,   Precision::I8,
            Precision::U16,  Precision::I16,
            Precision::U32,  Precision::I32,
            Precision::U64,  Precision::I64,
            Precision::BF16, Precision::FP16,
            Precision::FP32, Precision::FP64,
            Precision::BOOL
        };

        if (!supported_precisions.count(input_precision)) {
            IE_CPU_PLUGIN_THROW(NotImplemented)
                        << "Input image format " << input_precision << " is not supported yet...";
        }
    }

    auto config = orig_config;

    CNNNetwork clonedNetwork = InferenceEngine::details::cloneNetwork(network);
    const auto& lptProp = config.find(InferenceEngine::PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE);
    const bool enableLPT = (lptProp != config.end() && lptProp->second == PluginConfigParams::YES) /* enabled in the orig_config*/
            || Config::LPTransformsMode::On == engConfig.lpTransformsMode /* or already enabled for the plugin */;
    const auto& BF16Prop = config.find(InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16);
    bool enableBF16 = false;
    if (BF16Prop != config.end()) {
        if (BF16Prop->second == PluginConfigParams::YES) {
            enableBF16 = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core);
        } else {
            enableBF16 = false;
        }
    } else {
        enableBF16 = engConfig.enforceBF16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core);
    }
    const auto& dynamicBatchProp = config.find(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED);
    const bool enableDynamicBatch = (dynamicBatchProp != config.end() && dynamicBatchProp->second == PluginConfigParams::YES)
            || engConfig.enableDynamicBatch;

    auto snippetsMode = enableDynamicBatch ? Config::SnippetsMode::Disable : Config::SnippetsMode::Enable;
    const auto& snippetsModeProp = config.find(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE);
    if (snippetsMode == Config::SnippetsMode::Enable && snippetsModeProp != config.end()) {
        const auto& val = snippetsModeProp->second;
        if (val == PluginConfigInternalParams::IGNORE_CALLBACK)
            snippetsMode =  Config::SnippetsMode::IgnoreCallback;
        else if (val == PluginConfigInternalParams::DISABLE)
            snippetsMode =  Config::SnippetsMode::Disable;
        else
            IE_THROW() << "Wrong value for property key SNIPPETS_MODE. Expected values: ENABLE/DISABLE/IGNORE_CALLBACK";
    }

    auto nGraphFunc = clonedNetwork.getFunction();

    DEBUG_LOG(PrintableModel(*nGraphFunc, "org_"));

    Transformations transformations(nGraphFunc, enableLPT, enableBF16, isLegacyAPI(), snippetsMode, engConfig);
    transformations.UpToCpuSpecificOpSet();

    // need to check that all outputs have static shapes
    // checking that all inputs have static shapes is performed in the common part
    if (isLegacyAPI()) {
        for (const auto& res : nGraphFunc->get_results()) {
            if (res->get_input_partial_shape(0).is_dynamic()) {
                IE_THROW() << "CPU plug-in can't load a model with dynamic output shapes via legacy API.";
            }
        }
    }

    ApplyPerformanceHints(config, nGraphFunc);
    transformations.CpuSpecificOpSet();

    DEBUG_LOG(PrintableModel(*nGraphFunc, "cpu_"));

    // update the props after the perf mode translated to configs
    // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
    Config conf = engConfig;

    conf.readProperties(config);
    if (conf.enableDynamicBatch) {
        conf.batchLimit = static_cast<int>(network.getBatchSize());
    }

    // SSE runtime check is needed for some ATOM machine, which is x86-64 but w/o SSE
    static Xbyak::util::Cpu cpu;
    if (cpu.has(Xbyak::util::Cpu::tSSE)) {
        if (conf.denormalsOptMode == Config::DenormalsOptMode::DO_On) {
            flush_to_zero(true);
            denormals_as_zero(true);
        } else if (conf.denormalsOptMode == Config::DenormalsOptMode::DO_Off) {
            flush_to_zero(false);
            denormals_as_zero(false);
        }
    }

    return std::make_shared<ExecNetwork>(clonedNetwork, conf, extensionManager, shared_from_this());
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    streamsExplicitlySetForEngine = streamsSet(config);

    engConfig.readProperties(config);
}

bool Engine::isLegacyAPI() const {
    return !IsNewAPI();
}

Parameter Engine::GetConfigLegacy(const std::string& name, const std::map<std::string, Parameter>& options) const {
    Parameter result;
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        result = option->second;
    } else {
        IE_CPU_PLUGIN_THROW() << ". Unsupported config parameter: " << name;
    }
    return result;
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    if (isLegacyAPI())
        return GetConfigLegacy(name, options);

    if (name == ov::optimal_number_of_infer_requests) {
        const auto streams = engConfig.streamExecutorConfig._streams;
        return decltype(ov::optimal_number_of_infer_requests)::value_type(streams); // ov::optimal_number_of_infer_requests has no negative values
    } else if (name == ov::num_streams) {
        const auto streams = engConfig.streamExecutorConfig._streams;
        return decltype(ov::num_streams)::value_type(streams); // ov::num_streams has special negative values (AUTO = -1, NUMA = -2)
    } else if (name == ov::affinity) {
        const auto affinity = engConfig.streamExecutorConfig._threadBindingType;
        switch (affinity) {
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::NONE:
            return ov::Affinity::NONE;
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::CORES:
            return ov::Affinity::CORE;
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::NUMA:
            return ov::Affinity::NUMA;
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
            return ov::Affinity::HYBRID_AWARE;
        }
        return ov::Affinity::NONE;
    } else if (name == ov::inference_num_threads) {
        const auto num_threads = engConfig.streamExecutorConfig._threads;
        return decltype(ov::inference_num_threads)::value_type(num_threads);
    } else if (name == ov::enable_profiling.name()) {
        const bool perfCount = engConfig.collectPerfCounters;
        return decltype(ov::enable_profiling)::value_type(perfCount);
    } else if (name == ov::inference_precision) {
        const auto enforceBF16 = engConfig.enforceBF16;
        const auto inference_precision = enforceBF16 ? ov::element::bf16 : ov::element::f32;
        return decltype(ov::inference_precision)::value_type(inference_precision);
    } else if (name == ov::hint::performance_mode) {
        const auto perfHint = ov::util::from_string(engConfig.perfHintsConfig.ovPerfHint, ov::hint::performance_mode);
        return perfHint;
    } else if (name == ov::hint::num_requests) {
        const auto perfHintNumRequests = engConfig.perfHintsConfig.ovPerfHintNumRequests;
        return decltype(ov::hint::num_requests)::value_type(perfHintNumRequests);
    }
    /* Internally legacy parameters are used with new API as part of migration procedure.
     * This fallback can be removed as soon as migration completed */
    return GetConfigLegacy(name, options);
}

Parameter Engine::GetMetricLegacy(const std::string& name, const std::map<std::string, Parameter>& options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics = {
            METRIC_KEY(AVAILABLE_DEVICES),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(FULL_DEVICE_NAME),
            METRIC_KEY(OPTIMIZATION_CAPABILITIES),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS),
            METRIC_KEY(RANGE_FOR_STREAMS),
            METRIC_KEY(IMPORT_EXPORT_SUPPORT),
        };
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, deviceFullName);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { "" };
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities;
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16))
            capabilities.push_back(METRIC_VALUE(BF16));
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core))
            capabilities.push_back(METRIC_VALUE(WINOGRAD));
        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(FP16));
        capabilities.push_back(METRIC_VALUE(INT8));
        capabilities.push_back(METRIC_VALUE(BIN));
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && opt : engConfig._config)
            configKeys.push_back(opt.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 1, 1);
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, range);
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, parallel_get_max_threads());
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, range);
    } else if (name == METRIC_KEY(IMPORT_EXPORT_SUPPORT)) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    }

    IE_CPU_PLUGIN_THROW() << "Unsupported metric key: " << name;
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
    if (isLegacyAPI())
        return GetMetricLegacy(name, options);

    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };
    auto RW_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
    };

    if (name == ov::supported_properties) {
        std::vector<ov::PropertyName> roProperties {RO_property(ov::supported_properties.name()),
                                                    RO_property(ov::available_devices.name()),
                                                    RO_property(ov::range_for_async_infer_requests.name()),
                                                    RO_property(ov::range_for_streams.name()),
                                                    RO_property(ov::device::full_name.name()),
                                                    RO_property(ov::device::capabilities.name()),
                                                    RO_property(ov::caching_properties.name()),
        };
        // the whole config is RW before network is loaded.
        std::vector<ov::PropertyName> rwProperties {RW_property(ov::num_streams.name()),
                                                    RW_property(ov::affinity.name()),
                                                    RW_property(ov::inference_num_threads.name()),
                                                    RW_property(ov::enable_profiling.name()),
                                                    RW_property(ov::inference_precision.name()),
                                                    RW_property(ov::hint::performance_mode.name()),
                                                    RW_property(ov::hint::num_requests.name()),
        };

        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(roProperties.size() + rwProperties.size());
        supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
        supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

        return decltype(ov::supported_properties)::value_type(supportedProperties);
    } else if (name == ov::device::full_name) {
        return decltype(ov::device::full_name)::value_type(deviceFullName);
    } else if (name == ov::available_devices) {
        const std::vector<std::string> availableDevices = { "" };
        return decltype(ov::available_devices)::value_type(availableDevices);
    } else if (name == ov::device::capabilities) {
        std::vector<std::string> capabilities;
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16))
            capabilities.push_back(METRIC_VALUE(BF16));
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core))
            capabilities.push_back(METRIC_VALUE(WINOGRAD));
        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(FP16));
        capabilities.push_back(METRIC_VALUE(INT8));
        capabilities.push_back(METRIC_VALUE(BIN));
        capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
        return decltype(ov::device::capabilities)::value_type(capabilities);
    } else if (name == ov::range_for_async_infer_requests) {
        const std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 1, 1);
        return decltype(ov::range_for_async_infer_requests)::value_type(range);
    } else if (name == ov::range_for_streams) {
        const std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, parallel_get_max_threads());
        return decltype(ov::range_for_streams)::value_type(range);
    } else if (name == ov::caching_properties) {
        std::vector<ov::PropertyName> cachingProperties;
        return decltype(ov::caching_properties)::value_type(cachingProperties);
    }
    /* Internally legacy parameters are used with new API as part of migration procedure.
     * This fallback can be removed as soon as migration completed */
    return GetMetricLegacy(name, options);
}

void Engine::AddExtension(const InferenceEngine::IExtensionPtr& extension) {
    extensionManager->AddExtension(extension);
}

QueryNetworkResult Engine::QueryNetwork(const CNNNetwork& network, const std::map<std::string, std::string>& config) const {
    QueryNetworkResult res;

    WeightsSharing::Ptr fake_w_cache;

    // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
    Config conf = engConfig;
    conf.readProperties(config);

    if (conf.enableDynamicBatch) {
        conf.batchLimit = static_cast<int>(network.getBatchSize());
    }

    const auto& lptProp = config.find(InferenceEngine::PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE);
    const bool enableLPT = (lptProp != config.end() && lptProp->second == PluginConfigParams::YES) /* enabled in the orig_config*/
                        || Config::LPTransformsMode::On == engConfig.lpTransformsMode /* or already enabled */;

    auto snippetsMode = conf.enableDynamicBatch ? Config::SnippetsMode::Disable : Config::SnippetsMode::Enable;
    const auto& snippetsModeProp = config.find(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE);
    if (snippetsMode == Config::SnippetsMode::Enable && snippetsModeProp != config.end()) {
        const auto& val = snippetsModeProp->second;
        if (val == PluginConfigInternalParams::IGNORE_CALLBACK)
            snippetsMode =  Config::SnippetsMode::IgnoreCallback;
        else if (val == PluginConfigInternalParams::DISABLE)
            snippetsMode =  Config::SnippetsMode::Disable;
        else
            IE_THROW() << "Wrong value for property key SNIPPETS_MODE. Expected values: ENABLE/DISABLE/IGNORE_CALLBACK";
    }

    auto model = network.getFunction();
    if (model == nullptr) {
        IE_THROW() << "Only ngraph-based models are supported!";
    }

    auto context =
        std::make_shared<GraphContext>(conf, extensionManager, fake_w_cache, std::make_shared<std::mutex>(), false);

    auto supported = GetSupportedNodes(model,
                                       [&](std::shared_ptr<ov::Model>& model) {
                                           Transformations transformation(model, enableLPT, conf.enforceBF16, isLegacyAPI(), snippetsMode, engConfig);
                                           transformation.UpToCpuSpecificOpSet();
                                           transformation.CpuSpecificOpSet();
                                       },
                                       [&](const std::shared_ptr<ngraph::Node>& op) {
                                           std::unique_ptr<Node> ptr;
                                           try {
                                               ptr.reset(Node::factory().create(op, context));
                                           } catch (const InferenceEngine::Exception&) {
                                               return false;
                                           }
                                           return true;
                                       });

    for (auto&& layerName : supported) {
        res.supportedLayersMap.emplace(layerName, GetName());
    }

    return res;
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(std::istream& networkModel,
                                            const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "ImportNetwork");

    CNNNetworkDeserializer deserializer(networkModel,
        [this](const std::string& model, const Blob::CPtr& weights) {
            return GetCore()->ReadNetwork(model, weights, true);
        });

    CNNNetwork cnnnetwork;
    deserializer >> cnnnetwork;

    Config conf = engConfig;
    conf.readProperties(config);

    // import config props from caching model
    auto function = cnnnetwork.getFunction();
    if (function->has_rt_info("intel_cpu_hints_config") && !conf.perfHintsConfig.ovPerfHint.empty()) {
        const auto mode_name = conf.perfHintsConfig.ovPerfHint;
        if (mode_name == CONFIG_VALUE(LATENCY) || mode_name == CONFIG_VALUE(THROUGHPUT)) {
            const auto& hints_config = function->get_rt_info<ov::AnyMap>("intel_cpu_hints_config");
            const auto hints_param_name = mode_name + "_" + std::string(ov::num_streams.name());
            const auto it = hints_config.find(hints_param_name);
            if (it != hints_config.end()) {
                conf.readProperties({{std::string(ov::num_streams.name()), it->second.as<std::string>()}});
            } else {
                IE_THROW() << "Cache file doesn't contain precalculated number of streams for mode " << mode_name;
            }
        }
    }

    if (conf.enableDynamicBatch) {
        conf.batchLimit = static_cast<int>(cnnnetwork.getBatchSize());
    }

    auto execNetwork = std::make_shared<ExecNetwork>(cnnnetwork, conf, extensionManager, shared_from_this());

    execNetwork->setNetworkInputs(cnnnetwork.getInputsInfo());
    execNetwork->setNetworkOutputs(cnnnetwork.getOutputsInfo());
    SetExeNetworkInfo(execNetwork, cnnnetwork.getFunction());

    return execNetwork;
}

}   // namespace intel_cpu
}   // namespace ov

using namespace ov::intel_cpu;
static const Version version = {{2, 1}, CI_BUILD_NUMBER, "openvino_intel_cpu_plugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version)
