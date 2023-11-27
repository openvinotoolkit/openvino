// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"  // must be included first

#include "plugin.h"

#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "extension.h"
#include "extension_mngr.h"

#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "ie_system_conf.h"
#include "itt.h"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "performance_heuristics.hpp"
#include "serialize.h"
#include "transformations/transformation_pipeline.h"
#include "transformations/utils/utils.hpp"
#include "utils/denormals.hpp"
#include "weights_cache.hpp"

#if defined(__linux__)
# include <sys/auxv.h>
# include <signal.h>
# include <sys/mman.h>
#endif

#include <cpu/x64/cpu_isa_traits.hpp>

using namespace ov::threading;

#if defined(OV_CPU_WITH_ACL)
#include "nodes/executors/acl/acl_ie_scheduler.hpp"
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#endif

namespace ov {
namespace intel_cpu {

static std::string getDeviceFullName() {
    std::string brand_string;
#if defined(__EMSCRIPTEN__)
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

#if defined(OV_CPU_WITH_ACL)
std::mutex Engine::SchedulerGuard::mutex;
std::weak_ptr<Engine::SchedulerGuard> Engine::SchedulerGuard::ptr;

Engine::SchedulerGuard::SchedulerGuard() {
#if IE_THREAD == IE_THREAD_SEQ
    // To save state for ACL cores in single-thread mode
    arm_compute::Scheduler::set(arm_compute::Scheduler::Type::ST);
#else
    arm_compute::Scheduler::set(std::make_shared<ACLScheduler>());
#endif
}

std::shared_ptr<Engine::SchedulerGuard> Engine::SchedulerGuard::instance() {
    std::lock_guard<std::mutex> lock{SchedulerGuard::mutex};
    auto scheduler_guard_ptr = SchedulerGuard::ptr.lock();
    if (scheduler_guard_ptr == nullptr) {
        SchedulerGuard::ptr = scheduler_guard_ptr = std::make_shared<SchedulerGuard>();
    }
    return scheduler_guard_ptr;
}

Engine::SchedulerGuard::~SchedulerGuard() {
    // To save the state of scheduler after ACLScheduler has been executed
    // TODO: find out the cause of the state
    std::lock_guard<std::mutex> lock{this->dest_mutex};
    arm_compute::Scheduler::set(arm_compute::Scheduler::Type::ST);
}
#endif

Engine::Engine() :
    deviceFullName(getDeviceFullName()),
    specialSetup(new CPUSpecialSetup) {
    set_device_name("CPU");
    // Initialize Xbyak::util::Cpu object on Pcore for hybrid cores machine
    get_executor_manager()->execute_task_by_streams_executor(IStreamsExecutor::Config::PreferredCoreType::BIG, [] {
        dnnl::impl::cpu::x64::cpu();
    });
    extensionManager->AddExtension(std::make_shared<Extension>());
#if defined(OV_CPU_WITH_ACL)
    scheduler_guard = SchedulerGuard::instance();
#endif
}

Engine::~Engine() {
    executor_manager()->clear("CPU");
    executor_manager()->clear("CPUStreamsExecutor");
    executor_manager()->clear("CPUCallbackExecutor");
}

static bool streamsSet(const ov::AnyMap& config) {
    return config.count(InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS) ||
           config.count(ov::num_streams.name());
}

void Engine::apply_performance_hints(ov::AnyMap& config, const std::shared_ptr<ov::Model>& model) const {
    auto getNumStreamsLatency = [&]() {
        return std::pair<std::string, std::string>(CONFIG_VALUE(CPU_THROUGHPUT_NUMA),
                                                   ov::util::to_string(ov::streams::NUMA));
    };

    auto getNumStreamsThroughput = [&]() {
        const auto isa = dnnl::get_effective_cpu_isa();
        float isaSpecificThreshold = 1.0f;
        switch (isa) {
        case dnnl::cpu_isa::sse41:
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
        const float memThresholdAssumeLimitedForISA = ov::MemBandwidthPressure::LIMITED / isaSpecificThreshold;
        const float L2_cache_size = dnnl::utils::get_cache_size(2 /*level*/, true /*per core */);
        ov::MemBandwidthPressure networkToleranceForLowCache =
            ov::MemBandwidthPressureTolerance(model, L2_cache_size, memThresholdAssumeLimitedForISA);
        const auto default_streams = get_streams_num(engConfig.streamExecutorConfig._threadBindingType,
                                                   ov::threading::IStreamsExecutor::Config::StreamMode::DEFAULT,
                                                   engConfig.streamExecutorConfig._enable_hyper_thread);
        auto streams_info = default_streams;
        if (networkToleranceForLowCache.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
            if ((networkToleranceForLowCache.ratio_compute_convs == ov::MemBandwidthPressure::ALL) ||
                (networkToleranceForLowCache.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL)) {
                // all relevant layers (convs, etc) are compute-limited, the most aggressive val for #streams
                streams_info = get_streams_num(engConfig.streamExecutorConfig._threadBindingType,
                                             ov::threading::IStreamsExecutor::Config::StreamMode::AGGRESSIVE,
                                             engConfig.streamExecutorConfig._enable_hyper_thread);
            }  //  otherwise (no recognized layers) falling back to the default value
        } else if (networkToleranceForLowCache.max_mem_tolerance > memThresholdAssumeLimitedForISA) {
            // network is below the ISA-specific threshold
            streams_info = get_streams_num(engConfig.streamExecutorConfig._threadBindingType,
                                         ov::threading::IStreamsExecutor::Config::StreamMode::AGGRESSIVE,
                                         engConfig.streamExecutorConfig._enable_hyper_thread);
        } else if (networkToleranceForLowCache.max_mem_tolerance > ov::MemBandwidthPressure::LIMITED) {
            // network is below general threshold
            streams_info = get_streams_num(engConfig.streamExecutorConfig._threadBindingType,
                                         ov::threading::IStreamsExecutor::Config::StreamMode::LESSAGGRESSIVE,
                                         engConfig.streamExecutorConfig._enable_hyper_thread);
            streams_info.num_streams = std::max(default_streams.num_streams, streams_info.num_streams);
        }

        auto num_requests = config.find(ov::hint::num_requests.name());
        if (num_requests != config.end()) {  // arrived with config to the LoadNetwork (and thus higher pri)
            auto val = InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue(num_requests->second.as<std::string>());
            if (val > 0)
                streams_info.num_streams = std::min(streams_info.num_streams, val);
        } else if (engConfig.perfHintsConfig.ovPerfHintNumRequests) {  // set thru SetConfig to the plugin, 2nd priority
            streams_info.num_streams =
                std::min(streams_info.num_streams, engConfig.perfHintsConfig.ovPerfHintNumRequests);
        }
        return std::pair<std::string, StreamCfg>(std::to_string(streams_info.num_streams), streams_info);
    };

    OPENVINO_SUPPRESS_DEPRECATED_START
    auto getPerfHintName = [&]() {
        const bool streamsExplicitlySetForModel = streamsSet(config);
        // checking streams (to avoid overriding what user might explicitly set in the incoming config or previously via
        // SetConfig)
        if (streamsExplicitlySetForModel || streamsExplicitlySetForEngine)
            return std::string();

        const auto& perf_hint = config.find(ov::hint::performance_mode.name());
        // the perf_hint may have just arrived to the LoadNetwork, or was set with the plugin's SetConfig
        if (perf_hint == config.end() && engConfig.perfHintsConfig.ovPerfHint.empty())
            return std::string();
        /* performance hints set for network has higher pririty than engine ones.
         * This applies for all the configuration parameters */
        const auto perf_hint_name =
            (perf_hint != config.end())
                ? InferenceEngine::PerfHintsConfig::CheckPerformanceHintValue(perf_hint->second.as<std::string>())
                : engConfig.perfHintsConfig.ovPerfHint;
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
    model->set_rt_info(hints_props, "intel_cpu_hints_config");

    const auto perf_hint_name = getPerfHintName();
    if (perf_hint_name == CONFIG_VALUE(LATENCY)) {
        config[CONFIG_KEY(CPU_THROUGHPUT_STREAMS)] = latency_hints.first;
        config[ov::num_streams.name()] = latency_hints.second;
    } else if (perf_hint_name == CONFIG_VALUE(THROUGHPUT)) {
        config[CONFIG_KEY(CPU_THROUGHPUT_STREAMS)] = tput_hints.first;
        config[ov::num_streams.name()] = tput_hints.first;
        config[CONFIG_KEY_INTERNAL(BIG_CORE_STREAMS)] = std::to_string(tput_hints.second.big_core_streams);
        config[CONFIG_KEY_INTERNAL(SMALL_CORE_STREAMS)] = std::to_string(tput_hints.second.small_core_streams);
        config[CONFIG_KEY_INTERNAL(THREADS_PER_STREAM_BIG)] =
            std::to_string(tput_hints.second.threads_per_stream_big);
        config[CONFIG_KEY_INTERNAL(THREADS_PER_STREAM_SMALL)] =
            std::to_string(tput_hints.second.threads_per_stream_small);
        config[CONFIG_KEY_INTERNAL(SMALL_CORE_OFFSET)] = std::to_string(tput_hints.second.small_core_offset);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

void Engine::get_performance_streams(Config& config, const std::shared_ptr<ov::Model>& model) const{
    const auto perf_hint_name = config.perfHintsConfig.ovPerfHint;
    const int latency_streams = get_default_latency_streams(config.latencyThreadingMode);
    int streams;
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (config.streamExecutorConfig._streams_changed) {
        streams = config.streamExecutorConfig._streams;
    } else if (perf_hint_name == CONFIG_VALUE(LATENCY)) {
        streams = latency_streams;
    } else if (perf_hint_name == CONFIG_VALUE(THROUGHPUT)) {
        streams = 0;
    } else {
        streams = config.streamExecutorConfig._streams == 1 ? 0 : config.streamExecutorConfig._streams;
    }

    if (!((0 == config.streamExecutorConfig._streams) && config.streamExecutorConfig._streams_changed)) {
        get_num_streams(streams, model, config);
    }

    config._config[CONFIG_KEY(CPU_THROUGHPUT_STREAMS)] = std::to_string(config.streamExecutorConfig._streams);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

void Engine::calculate_streams(Config& conf, const std::shared_ptr<ov::Model>& model, bool imported) const{
    // import config props from caching model
    if (imported && !is_cpu_map_available()) {
        if (model->has_rt_info("intel_cpu_hints_config") && !conf.perfHintsConfig.ovPerfHint.empty()) {
            const auto mode_name = conf.perfHintsConfig.ovPerfHint;
            if (mode_name == CONFIG_VALUE(LATENCY) || mode_name == CONFIG_VALUE(THROUGHPUT)) {
                const auto& hints_config = model->get_rt_info<ov::AnyMap>("intel_cpu_hints_config");
                const auto hints_param_name = mode_name + "_" + std::string(ov::num_streams.name());
                const auto it = hints_config.find(hints_param_name);
                if (it != hints_config.end()) {
                    conf.readProperties({{std::string(ov::num_streams.name()), it->second.as<std::string>()}});
                } else {
                    OPENVINO_THROW("Cache file doesn't contain precalculated number of streams for mode ", mode_name);
                }
            }
        }
    }

    if (is_cpu_map_available()) {
        const auto model_prefer_name = std::string("MODEL_PREFER_THREADS");
        if (imported && model->has_rt_info("intel_cpu_hints_config")) {
            // load model_prefer_threads from cache
            int cache_model_prefer;
            const auto& hints_config = model->get_rt_info<ov::AnyMap>("intel_cpu_hints_config");
            const auto it_model_prefer = hints_config.find(model_prefer_name);
            if (it_model_prefer != hints_config.end()) {
                try {
                    cache_model_prefer = it_model_prefer->second.as<int>();
                } catch (const std::exception&) {
                    OPENVINO_THROW("Cache file doesn't have valid value for " + model_prefer_name);
                }

                conf.modelPreferThreads = cache_model_prefer;
            }
        }
        get_performance_streams(conf, model);
        // save model_prefer_threads to model rt_info when loading network
        if (!imported) {
            ov::AnyMap hints_props;
            hints_props.insert({model_prefer_name, std::to_string(conf.modelPreferThreads)});
            model->set_rt_info(hints_props, "intel_cpu_hints_config");
        }
    }
}

StreamCfg Engine::get_streams_num(ov::threading::IStreamsExecutor::ThreadBindingType thread_binding_type,
                                        int stream_mode,
                                        const bool enable_hyper_thread) const {
    const int sockets = static_cast<int>(get_available_numa_nodes().size());
    const int num_cores =
        thread_binding_type == IStreamsExecutor::ThreadBindingType::HYBRID_AWARE
            ? parallel_get_max_threads()
            : (sockets == 1 ? (enable_hyper_thread ? parallel_get_max_threads() :  get_number_of_cpu_cores())
                            : get_number_of_cpu_cores());
    const int num_cores_phy = get_number_of_cpu_cores();
    const int num_big_cores_phy = get_number_of_cpu_cores(true);
    const int num_small_cores = num_cores_phy - num_big_cores_phy;
    const int num_big_cores = num_cores > num_cores_phy ? num_big_cores_phy * 2 : num_big_cores_phy;
    StreamCfg stream_cfg = {0};

    if (stream_mode == IStreamsExecutor::Config::StreamMode::DEFAULT) {
        // bare minimum of streams (that evenly divides available number of core)
        if (thread_binding_type == IStreamsExecutor::ThreadBindingType::HYBRID_AWARE) {
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
    } else if (stream_mode == IStreamsExecutor::Config::StreamMode::AGGRESSIVE) {
        if (thread_binding_type == IStreamsExecutor::ThreadBindingType::HYBRID_AWARE) {
            stream_cfg.big_core_streams = num_big_cores;
            stream_cfg.small_core_streams = num_small_cores;
            stream_cfg.threads_per_stream_big = num_big_cores / stream_cfg.big_core_streams;
            stream_cfg.threads_per_stream_small =
                num_small_cores == 0 ? 0 : num_small_cores / stream_cfg.small_core_streams;
        } else {
            stream_cfg.num_streams = num_cores_phy;
        }
    } else if (stream_mode == IStreamsExecutor::Config::StreamMode::LESSAGGRESSIVE) {
        if (thread_binding_type == IStreamsExecutor::ThreadBindingType::HYBRID_AWARE) {
            stream_cfg.big_core_streams = num_big_cores / 2;
            stream_cfg.small_core_streams = num_small_cores / 2;
            stream_cfg.threads_per_stream_big = num_big_cores / stream_cfg.big_core_streams;
            stream_cfg.threads_per_stream_small =
                num_small_cores == 0 ? 0 : num_small_cores / stream_cfg.small_core_streams;
        } else {
            stream_cfg.num_streams = num_cores_phy / 2;
        }
    } else {
        OPENVINO_THROW("Wrong stream mode to get num of streams: ", stream_mode);
    }
    stream_cfg.num_streams = stream_cfg.num_streams > 0
                                 ? stream_cfg.num_streams
                                 : stream_cfg.big_core_streams + stream_cfg.small_core_streams;
    stream_cfg.small_core_offset = num_small_cores == 0 ? 0 : num_big_cores;
    return stream_cfg;
}

static bool shouldEnableLPT(const ov::AnyMap& modelConfig, const Config& engineConfig) {
    const auto& enableLPT = modelConfig.find(InferenceEngine::PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE);
    if (enableLPT == modelConfig.end()) // model config has higher priority
        return engineConfig.lpTransformsMode == Config::LPTransformsMode::On;

    const auto& val = enableLPT->second.as<std::string>();
    if (val == InferenceEngine::PluginConfigParams::YES)
        return true;
    else if (val == InferenceEngine::PluginConfigParams::NO)
        return false;
    else
        OPENVINO_THROW("Wrong value for property key LP_TRANSFORMS_MODE. Expected values: YES/NO");
}

static ov::element::Type getInferencePrecision(const ov::AnyMap& modelConfig,
                                               const Config& engineConfig,
                                               Config::ModelType modelType) {
    Config tempConf = engineConfig;
    tempConf.readProperties(modelConfig, modelType);
    return tempConf.inferencePrecision;
}

static Config::ModelType getModelType(const std::shared_ptr<const Model>& model) {
    return op::util::has_op_with_type<op::v1::Convolution>(model) ||
           op::util::has_op_with_type<op::v1::ConvolutionBackpropData>(model) ?
           Config::ModelType::CNN : Config::ModelType::Unknown;
}

static Config::SnippetsMode getSnippetsMode(const ov::AnyMap& modelConfig, const Config& engineConfig) {
    const auto& snippetsMode = modelConfig.find(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE);
    if (snippetsMode == modelConfig.end())    // not set explicitly
        return Config::SnippetsMode::Enable;  // enable by default

    const auto& val = snippetsMode->second.as<std::string>();
    if (val == InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK)
        return Config::SnippetsMode::IgnoreCallback;
    else if (val == InferenceEngine::PluginConfigInternalParams::DISABLE)
        return Config::SnippetsMode::Disable;
    else if (val == InferenceEngine::PluginConfigInternalParams::ENABLE)
        return Config::SnippetsMode::Enable;
    else
        OPENVINO_THROW("Wrong value for property key SNIPPETS_MODE. Expected values: ENABLE/DISABLE/IGNORE_CALLBACK");
}

std::shared_ptr<ov::ICompiledModel>
Engine::compile_model(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& orig_config) const{
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "Engine::compile_model");
    CREATE_DEBUG_TIMER(debugLoadTimer);

    // verification of supported input
    for (const auto &ii : model->inputs()) {
        auto input_precision = ii.get_element_type();
        static const std::set<ov::element::Type_t> supported_precisions = {ov::element::Type_t::u8,
                                                                           ov::element::Type_t::i8,
                                                                           ov::element::Type_t::u16,
                                                                           ov::element::Type_t::i16,
                                                                           ov::element::Type_t::u32,
                                                                           ov::element::Type_t::i32,
                                                                           ov::element::Type_t::u64,
                                                                           ov::element::Type_t::i64,
                                                                           ov::element::Type_t::bf16,
                                                                           ov::element::Type_t::f16,
                                                                           ov::element::Type_t::f32,
                                                                           ov::element::Type_t::f64,
                                                                           ov::element::Type_t::boolean};

        if (!supported_precisions.count(input_precision)) {
            OPENVINO_THROW_NOT_IMPLEMENTED("CPU plugin: Input image format ",
                                           input_precision,
                                           " is not supported yet...");
        }
    }

    auto config = orig_config;
    const std::shared_ptr<ov::Model> cloned_model = model->clone();
    const bool enableLPT = shouldEnableLPT(config, engConfig);
    Config::ModelType modelType = getModelType(model);
    ov::element::Type inferencePrecision = getInferencePrecision(config, engConfig, modelType);
    const Config::SnippetsMode snippetsMode = getSnippetsMode(config, engConfig);
    DEBUG_LOG(PrintableModel(*cloned_model, "org_"));

    // update the props after the perf mode translated to configs
    // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
    Config conf = engConfig;

    Transformations transformations(cloned_model, enableLPT, inferencePrecision, is_legacy_api(), snippetsMode, conf);
    transformations.UpToLpt();

    if (!is_cpu_map_available()) {
        apply_performance_hints(config, cloned_model);
    }

    conf.readProperties(config, modelType);
    calculate_streams(conf, cloned_model);

    transformations.PostLpt();
    transformations.Snippets();

    // need to check that all outputs have static shapes
    // checking that all inputs have static shapes is performed in the common part
    if (is_legacy_api()) {
        for (const auto& res : cloned_model->get_results()) {
            if (res->get_input_partial_shape(0).is_dynamic()) {
                OPENVINO_THROW("CPU plug-in can't load a model with dynamic output shapes via legacy API.");
            }
        }
    }

    transformations.CpuSpecificOpSet();
    DEBUG_LOG(PrintableModel(*cloned_model, "cpu_"));

    if ((cloned_model->inputs().size() != model->inputs().size()) ||
        (cloned_model->outputs().size() != model->outputs().size())) {
        OPENVINO_THROW("Input/output ports count mismatch between the original model and after the transformation! "
                       "Original model inputs count: ",
                       model->inputs().size(),
                       " after the transformations ",
                       cloned_model->inputs().size(),
                       ". Original model outputs count:",
                       model->inputs().size(),
                       " after the transformations ",
                       cloned_model->outputs().size());
    }
    // Make output ports have the same tensor names with original model
    for (size_t idx = 0; idx < cloned_model->outputs().size(); idx++) {
        auto new_result = cloned_model->output(idx);
        auto orig_result = model->output(idx);
        new_result.get_tensor().set_names(orig_result.get_tensor().get_names());
    }

    // SSE runtime check is needed for some ATOM machine, which is x86-64 but w/o SSE
    static Xbyak::util::Cpu cpu;
    if (cpu.has(Xbyak::util::Cpu::tSSE)) {
        if (conf.denormalsOptMode == Config::DenormalsOptMode::DO_On) {
            flush_to_zero(true);
            conf.DAZOn = denormals_as_zero(true);
        } else if (conf.denormalsOptMode == Config::DenormalsOptMode::DO_Off) {
            flush_to_zero(false);
            denormals_as_zero(false);
        }
    }
    return std::make_shared<CompiledModel>(cloned_model, shared_from_this(), conf, extensionManager);
}

void Engine::set_property(const ov::AnyMap &config) {
    // @todo after Legacy configuration is dropped, use some wrapper class to keep both the property and "ifSetExplicitly" flag
    streamsExplicitlySetForEngine = streamsSet(config);

    engConfig.readProperties(config);
}

bool Engine::is_legacy_api() const {
    return !get_core()->is_new_api();
}

ov::Any Engine::get_property_legacy(const std::string& name, const ov::AnyMap& options) const {
    ov::Any result;
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        result = option->second;
    } else {
        return get_metric_legacy(name, options);
    }
    return result;
}

ov::Any Engine::get_property(const std::string& name, const ov::AnyMap& options) const {
    if (is_legacy_api()) {
        auto ret = get_property_legacy(name, options);
        if (!ret.empty())
            return ret;
    }

    if (name == ov::optimal_number_of_infer_requests) {
        const auto streams = engConfig.streamExecutorConfig._streams;
        return decltype(ov::optimal_number_of_infer_requests)::value_type(
            streams);  // ov::optimal_number_of_infer_requests has no negative values
    } else if (name == ov::num_streams) {
        const auto streams = engConfig.streamExecutorConfig._streams;
        return decltype(ov::num_streams)::value_type(
            streams);  // ov::num_streams has special negative values (AUTO = -1, NUMA = -2)
    } else if (name == ov::affinity) {
        const auto affinity = engConfig.streamExecutorConfig._threadBindingType;
        switch (affinity) {
        case IStreamsExecutor::ThreadBindingType::NONE:
            return ov::Affinity::NONE;
        case IStreamsExecutor::ThreadBindingType::CORES:
            return ov::Affinity::CORE;
        case IStreamsExecutor::ThreadBindingType::NUMA:
            return ov::Affinity::NUMA;
        case IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
            return ov::Affinity::HYBRID_AWARE;
        }
        return ov::Affinity::NONE;
    } else if (name == ov::device::id.name()) {
        return decltype(ov::device::id)::value_type{engConfig.device_id};
    } else if (name == ov::inference_num_threads) {
        const auto num_threads = engConfig.streamExecutorConfig._threads;
        return decltype(ov::inference_num_threads)::value_type(num_threads);
    } else if (name == ov::enable_profiling.name()) {
        const bool perfCount = engConfig.collectPerfCounters;
        return decltype(ov::enable_profiling)::value_type(perfCount);
    } else if (name == ov::hint::inference_precision) {
        return decltype(ov::hint::inference_precision)::value_type(engConfig.inferencePrecision);
    } else if (name == ov::hint::performance_mode) {
        const auto perfHint = ov::util::from_string(engConfig.perfHintsConfig.ovPerfHint, ov::hint::performance_mode);
        return perfHint;
    } else if (name == ov::hint::enable_cpu_pinning) {
        const bool pin_value = engConfig.enableCpuPinning;
        return decltype(ov::hint::enable_cpu_pinning)::value_type(pin_value);
    } else if (name == ov::hint::scheduling_core_type) {
        const auto core_type = engConfig.schedulingCoreType;
        return core_type;
    } else if (name == ov::hint::enable_hyper_threading) {
        const bool ht_value = engConfig.enableHyperThreading;
        return decltype(ov::hint::enable_hyper_threading)::value_type(ht_value);
    } else if (name == ov::hint::num_requests) {
        const auto perfHintNumRequests = engConfig.perfHintsConfig.ovPerfHintNumRequests;
        return decltype(ov::hint::num_requests)::value_type(perfHintNumRequests);
    } else if (name == ov::hint::execution_mode) {
        return engConfig.executionMode;
    }
    return get_ro_property(name, options);
}

ov::Any Engine::get_metric_legacy(const std::string& name, const ov::AnyMap& options) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
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
    } else if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW}};
    } else if (name == ov::internal::caching_properties) {
        std::vector<ov::PropertyName> cachingProperties = { METRIC_KEY(FULL_DEVICE_NAME) };
        return decltype(ov::internal::caching_properties)::value_type(cachingProperties);
    }

    return {};
    OPENVINO_SUPPRESS_DEPRECATED_END
}

ov::Any Engine::get_ro_property(const std::string& name, const ov::AnyMap& options) const {
    if (is_legacy_api()) {
        ov::Any ret = get_metric_legacy(name, options);
        if (!ret.empty())
            return ret;
    }

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
        };
        // the whole config is RW before model is loaded.
        std::vector<ov::PropertyName> rwProperties {RW_property(ov::num_streams.name()),
                                                    RW_property(ov::affinity.name()),
                                                    RW_property(ov::inference_num_threads.name()),
                                                    RW_property(ov::enable_profiling.name()),
                                                    RW_property(ov::hint::inference_precision.name()),
                                                    RW_property(ov::hint::performance_mode.name()),
                                                    RW_property(ov::hint::execution_mode.name()),
                                                    RW_property(ov::hint::num_requests.name()),
                                                    RW_property(ov::hint::enable_cpu_pinning.name()),
                                                    RW_property(ov::hint::scheduling_core_type.name()),
                                                    RW_property(ov::hint::enable_hyper_threading.name()),
                                                    RW_property(ov::device::id.name()),
                                                    RW_property(ov::intel_cpu::denormals_optimization.name()),
                                                    RW_property(ov::intel_cpu::sparse_weights_decompression_rate.name()),
        };

        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(roProperties.size() + rwProperties.size());
        supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
        supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

        return decltype(ov::supported_properties)::value_type(supportedProperties);
    } else if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW}};
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
    } else if (name == ov::internal::caching_properties) {
        std::vector<ov::PropertyName> cachingProperties = { ov::device::full_name };
        return decltype(ov::internal::caching_properties)::value_type(cachingProperties);
    } else if (name == ov::intel_cpu::denormals_optimization) {
        return decltype(ov::intel_cpu::denormals_optimization)::value_type(engConfig.denormalsOptMode == Config::DenormalsOptMode::DO_On);
    } else if (name == ov::intel_cpu::sparse_weights_decompression_rate) {
        return decltype(ov::intel_cpu::sparse_weights_decompression_rate)::value_type(engConfig.fcSparseWeiDecompressionRate);
    }
    /* Internally legacy parameters are used with new API as part of migration procedure.
     * This fallback can be removed as soon as migration completed */
    auto ret = get_metric_legacy(name, options);
    if(!ret.empty())
        return ret;

    OPENVINO_THROW("Cannot get unsupport property: ", name);
}

OPENVINO_SUPPRESS_DEPRECATED_START
void Engine::add_extension(const InferenceEngine::IExtensionPtr& extension) {
    extensionManager->AddExtension(extension);
}
OPENVINO_SUPPRESS_DEPRECATED_END

ov::SupportedOpsMap Engine::query_model(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& config) const {
    WeightsSharing::Ptr fake_w_cache;

    if (model == nullptr) {
        OPENVINO_THROW("Only ngraph-based models are supported!");
    }

    Config conf = engConfig;
    Config::ModelType modelType = getModelType(model);
    conf.readProperties(config, modelType);

    const auto& lptProp = config.find(InferenceEngine::PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE);
    const bool enableLPT =
        (lptProp != config.end() &&
         lptProp->second.as<std::string>() == InferenceEngine::PluginConfigParams::YES) /* enabled in the orig_config*/
        || Config::LPTransformsMode::On == engConfig.lpTransformsMode /* or already enabled */;
    const Config::SnippetsMode snippetsMode = getSnippetsMode(config, conf);

    auto context =
        std::make_shared<GraphContext>(conf, extensionManager, fake_w_cache, false);

    auto supported = ov::get_supported_nodes(
        model,
        [&](std::shared_ptr<ov::Model>& model) {
            Transformations transformation(model,
                                           enableLPT,
                                           conf.inferencePrecision,
                                           is_legacy_api(),
                                           snippetsMode,
                                           engConfig);
            transformation.UpToLpt();
            transformation.PostLpt();
            transformation.Snippets();
            transformation.CpuSpecificOpSet();
        },
        [&](const std::shared_ptr<ov::Node>& op) {
            std::unique_ptr<Node> ptr;
            try {
                ptr.reset(Node::factory().create(op, context));
            } catch (const ov::Exception&) {
                return false;
            }
            return true;
        });

    ov::SupportedOpsMap res;
    for (auto&& layerName : supported) {
        res.emplace(layerName, get_device_name());
    }

    return res;
}

std::shared_ptr<ov::ICompiledModel> Engine::import_model(std::istream& networkModel,
                                            const ov::AnyMap& config) const{
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "import_model");

    ModelDeserializer deserializer(networkModel,
        [this](const std::string& model, const ov::Tensor& weights) {
            return get_core()->read_model(model, weights, true);
        });

    std::shared_ptr<ov::Model> model;
    deserializer >> model;

    Config conf = engConfig;
    Config::ModelType modelType = getModelType(model);
    conf.readProperties(config, modelType);

    // import config props from caching model
    calculate_streams(conf, model, true);

    auto compiled_model = std::make_shared<CompiledModel>(model, shared_from_this(), conf, extensionManager, true);
    return compiled_model;
}
}   // namespace intel_cpu
}   // namespace ov

using namespace ov::intel_cpu;

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_arm_cpu_plugin"};
#elif defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_intel_cpu_plugin"};
#elif defined(OPENVINO_ARCH_RISCV64)
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_riscv_cpu_plugin"};
#else
#error "Undefined system processor"
#endif

OV_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version)
