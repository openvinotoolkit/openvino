// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.h"

#include "internal_properties.hpp"
#include "itt.h"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "serialize.h"
#include "transformations/transformation_pipeline.h"
#include "transformations/utils/utils.hpp"
#include "utils/denormals.hpp"
#include "utils/precision_support.h"
#include "weights_cache.hpp"

#if defined(__linux__)
#    include <signal.h>
#    include <sys/auxv.h>
#    include <sys/mman.h>
#endif

#include "cpu/x64/cpu_isa_traits.hpp"

using namespace ov::threading;

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
    const unsigned int addr_list[3] = {0x80000002, 0x80000003, 0x80000004};
    unsigned int regs[4];
    for (auto addr : addr_list) {
        regs[0] = addr;
#    ifdef _WIN32
        __cpuid(reinterpret_cast<int*>(regs), regs[0]);
#    else
        __cpuid(regs[0], regs[0], regs[1], regs[2], regs[3]);
#    endif
        char* ch = reinterpret_cast<char*>(&regs[0]);
        for (size_t j = 0; j < sizeof(regs); j++)
            if (ch[j] != '\0')
                brand_string += ch[j];
    }
#else
#    error "Unkown CPU architecture. Please, add support to openvino/core/visibility.hpp"
#endif
    return brand_string;
}

#if defined(__linux__)

#    ifndef AT_MINSIGSTKSZ
#        define AT_MINSIGSTKSZ 51
#    endif

class SigAltStackSetup {
    stack_t new_stack{0};
    stack_t old_stack{0};

public:
    SigAltStackSetup() {
        memset(&old_stack, 0, sizeof(old_stack));
        memset(&new_stack, 0, sizeof(new_stack));

        auto minsigstksz = getauxval(AT_MINSIGSTKSZ);
        auto new_size = minsigstksz + SIGSTKSZ;
        void* altstack = mmap(NULL, new_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
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
#else   // __linux__
class CPUSpecialSetup {
public:
    CPUSpecialSetup() = default;
};
#endif  // __linux__

Plugin::Plugin() : deviceFullName(getDeviceFullName()), specialSetup(new CPUSpecialSetup) {
    set_device_name("CPU");
    // Initialize Xbyak::util::Cpu object on Pcore for hybrid cores machine
    get_executor_manager()->execute_task_by_streams_executor(ov::hint::SchedulingCoreType::PCORE_ONLY, [] {
        dnnl::impl::cpu::x64::cpu();
    });
    auto& ov_version = ov::get_openvino_version();
    m_compiled_model_runtime_properties["OV_VERSION"] = std::string(ov_version.buildNumber);
}

Plugin::~Plugin() {
    executor_manager()->clear("CPU");
    executor_manager()->clear("CPUStreamsExecutor");
    executor_manager()->clear("CPUCallbackExecutor");
}

static bool streamsSet(const ov::AnyMap& config) {
    return config.count(ov::num_streams.name());
}

void Plugin::get_performance_streams(Config& config, const std::shared_ptr<ov::Model>& model) const {
    int streams_set = config.streams;
    int streams;
    if (config.streamsChanged) {
        streams = streams_set;
    } else if (config.hintPerfMode == ov::hint::PerformanceMode::LATENCY) {
        streams = 1;
    } else if (config.hintPerfMode == ov::hint::PerformanceMode::THROUGHPUT) {
        streams = 0;
    } else {
        streams = streams_set == 1 ? 0 : streams_set;
    }

    if (!((0 == streams_set) && config.streamsChanged)) {
        get_num_streams(streams, model, config);
    } else {
        config.streamExecutorConfig = IStreamsExecutor::Config{"CPUStreamsExecutor", streams};
    }
}

void Plugin::calculate_streams(Config& conf, const std::shared_ptr<ov::Model>& model, bool imported) const {
    const auto model_prefer_name = std::string("MODEL_PREFER_THREADS");
    if (imported && model->has_rt_info("intel_cpu_hints_config")) {
        // load model_prefer_threads from cache
        int cache_model_prefer;
        const auto& hints_config = model->get_rt_info<ov::AnyMap>("intel_cpu_hints_config");
        const auto it_model_prefer = hints_config.find(model_prefer_name);
        if (it_model_prefer != hints_config.end()) {
            try {
                cache_model_prefer = it_model_prefer->second.as<int>();
            } catch (const ov::Exception&) {
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

static bool shouldEnableLPT(const ov::AnyMap& modelConfig, const Config& engineConfig) {
    const auto& enableLPT = modelConfig.find(ov::intel_cpu::lp_transforms_mode.name());
    if (enableLPT == modelConfig.end())  // model config has higher priority
        return engineConfig.lpTransformsMode == Config::LPTransformsMode::On;

    try {
        return enableLPT->second.as<bool>();
    } catch (ov::Exception&) {
        OPENVINO_THROW("Wrong value ",
                       enableLPT->second.as<std::string>(),
                       " for property key LP_TRANSFORMS_MODE. Expected values: YES/NO");
    }
}

static ov::element::Type getInferencePrecision(const ov::AnyMap& modelConfig,
                                               const Config& engineConfig,
                                               Config::ModelType modelType) {
    Config tempConf = engineConfig;
    tempConf.readProperties(modelConfig, modelType);
    return tempConf.inferencePrecision;
}

static Config::ModelType getModelType(const std::shared_ptr<const Model>& model) {
    if (op::util::has_op_with_type<op::v1::Convolution>(model) ||
        op::util::has_op_with_type<op::v1::ConvolutionBackpropData>(model))
        return Config::ModelType::CNN;
    
    if (op::util::has_op_with_type<op::v13::ScaledDotProductAttention>(model) &&
        model->get_variables().size() > 0)
        return Config::ModelType::LLM;

    return Config::ModelType::Unknown;
}

static Config::SnippetsMode getSnippetsMode(const ov::AnyMap& modelConfig, const Config& engineConfig) {
    const auto& snippetsMode = modelConfig.find(ov::intel_cpu::snippets_mode.name());
    if (snippetsMode == modelConfig.end())    // not set explicitly
        return Config::SnippetsMode::Enable;  // enable by default

    const auto& val = snippetsMode->second.as<std::string>();
    if (val == ov::util::to_string(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK))
        return Config::SnippetsMode::IgnoreCallback;
    else if (val == ov::util::to_string(ov::intel_cpu::SnippetsMode::DISABLE))
        return Config::SnippetsMode::Disable;
    else if (val == ov::util::to_string(ov::intel_cpu::SnippetsMode::ENABLE))
        return Config::SnippetsMode::Enable;
    else
        OPENVINO_THROW("Wrong value for property key SNIPPETS_MODE. Expected values: ENABLE/DISABLE/IGNORE_CALLBACK");
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& orig_config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "Plugin::compile_model");
    CREATE_DEBUG_TIMER(debugLoadTimer);

    // verification of supported input
    for (const auto& ii : model->inputs()) {
        auto input_precision = ii.get_element_type();
        static const std::set<ov::element::Type_t> supported_precisions = {ov::element::Type_t::u4,
                                                                           ov::element::Type_t::i4,
                                                                           ov::element::Type_t::u8,
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
                                                                           ov::element::Type_t::boolean,
                                                                           ov::element::Type_t::string};

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

    Transformations transformations(cloned_model, enableLPT, inferencePrecision, snippetsMode, conf);

    transformations.UpToLpt();

    conf.readProperties(config, modelType);
    calculate_streams(conf, cloned_model);

    if (conf.streamExecutorConfig.get_sub_stream_mode() ==
        IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_FOR_SOCKET) {
        int num_sub_streams = conf.streamExecutorConfig.get_sub_streams();
        transformations.SetSubStreamNum(num_sub_streams);
    }

    transformations.PostLpt();
    transformations.Snippets();

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
    return std::make_shared<CompiledModel>(cloned_model, shared_from_this(), conf, false);
}

void Plugin::set_property(const ov::AnyMap& config) {
    // @todo after Legacy configuration is dropped, use some wrapper class to keep both the property and
    // "ifSetExplicitly" flag
    streamsExplicitlySetForEngine = streamsSet(config);

    engConfig.readProperties(config);
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& options) const {
    if (name == ov::optimal_number_of_infer_requests) {
        const auto streams = engConfig.streamExecutorConfig.get_streams();
        return decltype(ov::optimal_number_of_infer_requests)::value_type(
            streams);  // ov::optimal_number_of_infer_requests has no negative values
    } else if (name == ov::num_streams) {
        const auto streams = engConfig.streamExecutorConfig.get_streams();
        return decltype(ov::num_streams)::value_type(
            streams);  // ov::num_streams has special negative values (AUTO = -1, NUMA = -2)
        OPENVINO_SUPPRESS_DEPRECATED_START
    } else if (name == ov::affinity) {
        const auto affinity = engConfig.threadBindingType;
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
        OPENVINO_SUPPRESS_DEPRECATED_END
    } else if (name == ov::device::id.name()) {
        return decltype(ov::device::id)::value_type{engConfig.device_id};
    } else if (name == ov::inference_num_threads) {
        const auto threads = engConfig.streamExecutorConfig.get_threads();
        return decltype(ov::inference_num_threads)::value_type(threads);
    } else if (name == ov::enable_profiling.name()) {
        const bool perfCount = engConfig.collectPerfCounters;
        return decltype(ov::enable_profiling)::value_type(perfCount);
    } else if (name == ov::hint::inference_precision) {
        return decltype(ov::hint::inference_precision)::value_type(engConfig.inferencePrecision);
    } else if (name == ov::hint::performance_mode) {
        return engConfig.hintPerfMode;
    } else if (name == ov::hint::enable_cpu_pinning) {
        const bool pin_value = engConfig.enableCpuPinning;
        return decltype(ov::hint::enable_cpu_pinning)::value_type(pin_value);
    } else if (name == ov::hint::scheduling_core_type) {
        const auto core_type = engConfig.schedulingCoreType;
        return core_type;
    } else if (name == ov::hint::model_distribution_policy) {
        const auto& distribution_policy = engConfig.modelDistributionPolicy;
        return distribution_policy;
    } else if (name == ov::hint::enable_hyper_threading) {
        const bool ht_value = engConfig.enableHyperThreading;
        return decltype(ov::hint::enable_hyper_threading)::value_type(ht_value);
    } else if (name == ov::hint::num_requests) {
        return decltype(ov::hint::num_requests)::value_type(engConfig.hintNumRequests);
    } else if (name == ov::hint::execution_mode) {
        return engConfig.executionMode;
    } else if (name == ov::internal::compiled_model_runtime_properties.name()) {
        auto model_runtime_properties = ov::Any(m_compiled_model_runtime_properties);
        return decltype(ov::internal::compiled_model_runtime_properties)::value_type(
            std::move(model_runtime_properties.as<std::string>()));
    } else if (name == ov::log::level) {
        return engConfig.logLevel;
    } else if (name == ov::internal::compiled_model_runtime_properties_supported.name()) {
        ov::Any res = true;
        auto it = options.find(ov::internal::compiled_model_runtime_properties.name());
        if (it == options.end()) {
            res = false;
        } else {
            ov::AnyMap input_map = it->second.as<ov::AnyMap>();
            for (auto& item : m_compiled_model_runtime_properties) {
                auto it = input_map.find(item.first);
                if (it == input_map.end() || it->second.as<std::string>() != item.second.as<std::string>()) {
                    res = false;
                    break;
                }
            }
        }
        return res;
    } else if (name == ov::internal::exclusive_async_requests.name()) {
        return engConfig.exclusiveAsyncRequests;
    } else if (name == ov::hint::dynamic_quantization_group_size) {
        return decltype(ov::hint::dynamic_quantization_group_size)::value_type(
            engConfig.fcDynamicQuantizationGroupSize);
    } else if (name == ov::hint::kv_cache_precision) {
        return decltype(ov::hint::kv_cache_precision)::value_type(engConfig.kvCachePrecision);
    }
    return get_ro_property(name, options);
}

ov::Any Plugin::get_ro_property(const std::string& name, const ov::AnyMap& options) const {
    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };
    auto RW_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
    };

    if (name == ov::supported_properties) {
        std::vector<ov::PropertyName> roProperties{
            RO_property(ov::supported_properties.name()),
            RO_property(ov::available_devices.name()),
            RO_property(ov::range_for_async_infer_requests.name()),
            RO_property(ov::range_for_streams.name()),
            RO_property(ov::execution_devices.name()),
            RO_property(ov::device::full_name.name()),
            RO_property(ov::device::capabilities.name()),
            RO_property(ov::device::type.name()),
            RO_property(ov::device::architecture.name()),
        };
        // the whole config is RW before model is loaded.
        std::vector<ov::PropertyName> rwProperties{
            RW_property(ov::num_streams.name()),
            RW_property(ov::inference_num_threads.name()),
            RW_property(ov::enable_profiling.name()),
            RW_property(ov::hint::inference_precision.name()),
            RW_property(ov::hint::performance_mode.name()),
            RW_property(ov::hint::execution_mode.name()),
            RW_property(ov::hint::num_requests.name()),
            RW_property(ov::hint::enable_cpu_pinning.name()),
            RW_property(ov::hint::scheduling_core_type.name()),
            RW_property(ov::hint::model_distribution_policy.name()),
            RW_property(ov::hint::enable_hyper_threading.name()),
            RW_property(ov::device::id.name()),
            RW_property(ov::intel_cpu::denormals_optimization.name()),
            RW_property(ov::log::level.name()),
            RW_property(ov::intel_cpu::sparse_weights_decompression_rate.name()),
            RW_property(ov::hint::dynamic_quantization_group_size.name()),
            RW_property(ov::hint::kv_cache_precision.name()),
        };

        OPENVINO_SUPPRESS_DEPRECATED_START
        rwProperties.insert(rwProperties.end(), RW_property(ov::affinity.name()));
        OPENVINO_SUPPRESS_DEPRECATED_END

        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(roProperties.size() + rwProperties.size());
        supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
        supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

        return decltype(ov::supported_properties)::value_type(std::move(supportedProperties));
    } else if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::compiled_model_runtime_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::compiled_model_runtime_properties_supported.name(),
                             ov::PropertyMutability::RO}};
    } else if (name == ov::device::full_name) {
        return decltype(ov::device::full_name)::value_type(deviceFullName);
    } else if (name == ov::available_devices) {
        const std::vector<std::string> availableDevices = {""};
        return decltype(ov::available_devices)::value_type(availableDevices);
    } else if (name == ov::device::capabilities) {
        std::vector<std::string> capabilities;
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16) ||
            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2))
            capabilities.push_back(ov::device::capability::BF16);
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core))
            capabilities.push_back(ov::device::capability::WINOGRAD);
        capabilities.push_back(ov::device::capability::FP32);
        if (hasHardwareSupport(ov::element::f16))
            capabilities.push_back(ov::device::capability::FP16);
        capabilities.push_back(ov::device::capability::INT8);
        capabilities.push_back(ov::device::capability::BIN);
        capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
        return decltype(ov::device::capabilities)::value_type(std::move(capabilities));
    } else if (name == ov::range_for_async_infer_requests) {
        const std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 1, 1);
        return decltype(ov::range_for_async_infer_requests)::value_type(range);
    } else if (name == ov::range_for_streams) {
        const std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, parallel_get_max_threads());
        return decltype(ov::range_for_streams)::value_type(range);
    } else if (name == ov::internal::caching_properties) {
        std::vector<ov::PropertyName> cachingProperties = {ov::device::full_name};
        return decltype(ov::internal::caching_properties)::value_type(std::move(cachingProperties));
    } else if (name == ov::intel_cpu::denormals_optimization) {
        return decltype(ov::intel_cpu::denormals_optimization)::value_type(engConfig.denormalsOptMode ==
                                                                           Config::DenormalsOptMode::DO_On);
    } else if (name == ov::intel_cpu::sparse_weights_decompression_rate) {
        return decltype(ov::intel_cpu::sparse_weights_decompression_rate)::value_type(
            engConfig.fcSparseWeiDecompressionRate);
    } else if (name == ov::execution_devices) {
        return decltype(ov::execution_devices)::value_type{get_device_name()};
    } else if (name == ov::device::type) {
        return decltype(ov::device::type)::value_type(ov::device::Type::INTEGRATED);
    } else if (name == ov::device::architecture) {
#if defined(OPENVINO_ARCH_X86_64)
        return decltype(ov::device::architecture)::value_type{"intel64"};
#elif defined(OPENVINO_ARCH_X86)
        return decltype(ov::device::architecture)::value_type{"ia32"};
#elif defined(OPENVINO_ARCH_ARM)
        return decltype(ov::device::architecture)::value_type{"armhf"};
#elif defined(OPENVINO_ARCH_ARM64)
        return decltype(ov::device::architecture)::value_type{"arm64"};
#elif defined(OPENVINO_ARCH_RISCV64)
        return decltype(ov::device::architecture)::value_type{"riscv"};
#else
#    error "Undefined system processor"
#endif
    }

    OPENVINO_THROW("Cannot get unsupported property: ", name);
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& config) const {
    WeightsSharing::Ptr fake_w_cache;

    if (model == nullptr) {
        OPENVINO_THROW("Only ngraph-based models are supported!");
    }

    Config conf = engConfig;
    Config::ModelType modelType = getModelType(model);
    conf.readProperties(config, modelType);

    const auto& lptProp = config.find(ov::intel_cpu::lp_transforms_mode.name());
    const bool enableLPT =
        (lptProp != config.end() && lptProp->second.as<bool>() == true) /* enabled in the orig_config*/
        || Config::LPTransformsMode::On == engConfig.lpTransformsMode /* or already enabled */;
    const Config::SnippetsMode snippetsMode = getSnippetsMode(config, conf);

    auto context = std::make_shared<GraphContext>(conf, fake_w_cache, false);

    auto supported = ov::get_supported_nodes(
        model,
        [&](std::shared_ptr<ov::Model>& model) {
            Transformations transformation(model, enableLPT, conf.inferencePrecision, snippetsMode, engConfig);
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

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& networkModel, const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "import_model");

    ModelDeserializer deserializer(networkModel, [this](const std::string& model, const ov::Tensor& weights) {
        return get_core()->read_model(model, weights, true);
    });

    std::shared_ptr<ov::Model> model;
    deserializer >> model;

    Config conf = engConfig;
    Config::ModelType modelType = getModelType(model);

    // check ov::loaded_from_cache property and erase it to avoid exception in readProperties.
    auto _config = config;
    const auto& it = _config.find(ov::loaded_from_cache.name());
    bool loaded_from_cache = false;
    if (it != _config.end()) {
        loaded_from_cache = it->second.as<bool>();
        _config.erase(it);
    }
    conf.readProperties(_config, modelType);

    // import config props from caching model
    calculate_streams(conf, model, true);
    auto compiled_model = std::make_shared<CompiledModel>(model, shared_from_this(), conf, loaded_from_cache);
    return compiled_model;
}
}  // namespace intel_cpu
}  // namespace ov

using namespace ov::intel_cpu;

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_arm_cpu_plugin"};
#elif defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_intel_cpu_plugin"};
#elif defined(OPENVINO_ARCH_RISCV64)
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_riscv_cpu_plugin"};
#else
#    error "Undefined system processor"
#endif

OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)
