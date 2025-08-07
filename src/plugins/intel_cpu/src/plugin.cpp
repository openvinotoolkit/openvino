// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.h"

#include <cstddef>
#include <cstring>
#include <fstream>
#include <istream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

#include "compiled_model.h"
#include "config.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_streams_calculation.hpp"
#include "graph_context.h"
#include "internal_properties.hpp"
#include "itt.h"
#include "node.h"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/version.hpp"
#include "openvino/itt.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "sigstack_manager.h"
#include "transformations/transformation_pipeline.h"
#include "transformations/utils/utils.hpp"
#include "utils/codec_xor.hpp"
#include "utils/debug_capabilities.h"
#include "utils/denormals.hpp"
#include "utils/precision_support.h"
#include "utils/serialize.hpp"
#include "weights_cache.hpp"
#include "xbyak/xbyak_util.h"

using namespace ov::threading;

namespace ov::intel_cpu {

static std::string getDeviceFullName() {
    std::string brand_string;
#if defined(__EMSCRIPTEN__)
    brand_string = "WebAssembly CPU";
#elif defined(OPENVINO_ARCH_RISCV64)
    // TODO: extract actual device name
    brand_string = "RISCV-64 CPU";
#elif defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
#    if defined(__APPLE__) || defined(__MACOSX)
    {
        auto read_sysctl_str = [](const char* name) -> std::string {
            size_t size = 0;
            if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
                return {};
            }
            std::string out(size, '\0');
            if (sysctlbyname(name, out.data(), &size, nullptr, 0) != 0 || size == 0) {
                return {};
            }
            if (!out.empty() && out.back() == '\0') {
                out.pop_back();
            }
            return out;
        };

        brand_string = read_sysctl_str("machdep.cpu.brand_string");
        if (brand_string.empty()) {
            brand_string = read_sysctl_str("hw.model");
        }
    }
#    elif defined(__linux__)
    {
        auto trim = [](std::string s) -> std::string {
            const auto start = s.find_first_not_of(" \t\r\n");
            const auto end = s.find_last_not_of(" \t\r\n");
            if (start == std::string::npos || end == std::string::npos) {
                return {};
            }
            return s.substr(start, end - start + 1);
        };
        auto read_first_line = [&](const char* path) -> std::string {
            std::ifstream f(path);
            if (!f.is_open()) {
                return {};
            }
            std::string line;
            std::getline(f, line);
            return trim(line);
        };
        auto pick_value = [&](const std::string& s) -> std::string {
            const auto pos = s.find(':');
            if (pos == std::string::npos) {
                return {};
            }
            return trim(s.substr(pos + 1));
        };

        // 1) Prefer device-tree model if available (not present in many containers)
        brand_string = read_first_line("/sys/firmware/devicetree/base/model");
        if (brand_string.empty()) {
            brand_string = read_first_line("/proc/device-tree/model");
        }

        // 2) Fall back to /proc/cpuinfo keys commonly seen on ARM
        if (brand_string.empty()) {
            std::ifstream cpuinfo("/proc/cpuinfo");
            std::string line;
            std::string implementer_hex;  // e.g., 0x41
            std::string part_hex;         // e.g., 0xd40
            while (cpuinfo.is_open() && std::getline(cpuinfo, line)) {
                if (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0 ||
                    line.rfind("Processor", 0) == 0 || line.rfind("Model", 0) == 0) {
                    auto v = pick_value(line);
                    if (!v.empty()) {
                        brand_string = v;
                        break;
                    }
                } else if (line.rfind("CPU implementer", 0) == 0) {
                    implementer_hex = pick_value(line);
                } else if (line.rfind("CPU part", 0) == 0) {
                    part_hex = pick_value(line);
                }
            }

            // 3) If we still don't have a friendly string, synthesize something readable
            if (brand_string.empty()) {
                auto vendor_from_impl = [](const std::string& hex) -> const char* {
                    // Map common implementer IDs (see Linux arch/arm64/include/asm/sysreg.h / MIDR)
                    if (hex.length() >= 3) {
                        auto id = std::stoul(hex, nullptr, 16);
                        switch (id) {
                        case 0x41:
                            return "ARM";
                        case 0x42:
                            return "Broadcom";
                        case 0x43:
                            return "Cavium";
                        case 0x44:
                            return "DEC";
                        case 0x46:
                            return "Fujitsu";
                        case 0x48:
                            return "HiSilicon";
                        case 0x49:
                            return "Infineon";
                        case 0x4D:
                            return "Motorola";
                        case 0x4E:
                            return "NVIDIA";
                        case 0x50:
                            return "Applied Micro";
                        case 0x51:
                            return "Qualcomm";
                        case 0x53:
                            return "Samsung";
                        case 0x56:
                            return "Marvell";
                        case 0x61:
                            return "Apple";
                        case 0x69:
                            return "Intel";
                        case 0xC0:
                            return "Ampere";
                        default:
                            return nullptr;
                        }
                    }
                    return nullptr;
                };

                const char* vendor = vendor_from_impl(implementer_hex);
                if (vendor) {
                    if (!part_hex.empty()) {
                        brand_string = std::string(vendor) + " (" + part_hex + ")";
                    } else {
                        brand_string = vendor;
                    }
                }
            }
        }
    }
#    endif
    if (brand_string.empty()) {
        brand_string = "ARM CPU";
    }
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
        auto* ch = reinterpret_cast<char*>(&regs[0]);
        for (size_t j = 0; j < sizeof(regs); j++) {
            if (ch[j] != '\0') {
                brand_string += ch[j];
            }
        }
    }
#else
#    error "Unkown CPU architecture. Please, add support to openvino/core/visibility.hpp"
#endif
    return brand_string;
}

Plugin::Plugin() : deviceFullName(getDeviceFullName()), specialSetup(new CPUSpecialSetup) {
    set_device_name("CPU");
    // Initialize Xbyak::util::Cpu object on Pcore for hybrid cores machine
    get_executor_manager()->execute_task_by_streams_executor(ov::hint::SchedulingCoreType::PCORE_ONLY, [] {
        dnnl::impl::cpu::x64::cpu();
    });
    const auto& ov_version = ov::get_openvino_version();
    m_compiled_model_runtime_properties["OV_VERSION"] = std::string(ov_version.buildNumber);
    m_msg_manager = ov::threading::message_manager();
}

Plugin::~Plugin() {
    executor_manager()->clear("CPU");
    executor_manager()->clear("CPUStreamsExecutor");
    executor_manager()->clear("CPUMainStreamExecutor");
    executor_manager()->clear("CPUCallbackExecutor");
}

static bool streamsSet(const ov::AnyMap& config) {
    return config.find(ov::num_streams.name()) != config.end();
}

void Plugin::get_performance_streams(Config& config, const std::shared_ptr<ov::Model>& model) {
    int streams_set = config.streams;
    int streams = 0;
    if (config.streamsChanged) {
        streams = streams_set;
    } else if (config.hintPerfMode == ov::hint::PerformanceMode::LATENCY) {
        streams = 1;
    } else if (config.hintPerfMode == ov::hint::PerformanceMode::THROUGHPUT) {
        streams = 0;
    } else {
        streams = streams_set == 1 ? 0 : streams_set;
    }

    if ((0 != streams_set) || !config.streamsChanged) {
        get_num_streams(streams, model, config);
    } else {
        config.streamExecutorConfig = IStreamsExecutor::Config{"CPUStreamsExecutor", streams};
    }
}

void Plugin::calculate_streams(Config& conf, const std::shared_ptr<ov::Model>& model, bool imported) {
    const auto model_prefer_name = std::string("MODEL_PREFER_THREADS");
    if (imported && model->has_rt_info("intel_cpu_hints_config")) {
        // load model_prefer_threads from cache
        int cache_model_prefer = 0;
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

static Config::ModelType getModelType(const std::shared_ptr<const Model>& model) {
    if (op::util::has_op_with_type<op::v1::Convolution>(model) ||
        op::util::has_op_with_type<op::v1::ConvolutionBackpropData>(model)) {
        return Config::ModelType::CNN;
    }

    if ((op::util::has_op_with_type<op::v13::ScaledDotProductAttention>(model) && !model->get_variables().empty()) ||
        op::util::has_op_with_type<ov::op::PagedAttentionExtension>(model)) {
        return Config::ModelType::LLM;
    }

    return Config::ModelType::Unknown;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& orig_config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "Plugin::compile_model");
    CREATE_DEBUG_TIMER(debugLoadTimer);

    // verification of supported input
    for (const auto& ii : model->inputs()) {
        auto input_precision = ii.get_element_type();
        static const std::set<ov::element::Type_t> supported_precisions = {
            ov::element::Type_t::u4,     ov::element::Type_t::i4,      ov::element::Type_t::u8,
            ov::element::Type_t::i8,     ov::element::Type_t::f8e4m3,  ov::element::Type_t::f8e5m2,
            ov::element::Type_t::u16,    ov::element::Type_t::i16,     ov::element::Type_t::u32,
            ov::element::Type_t::i32,    ov::element::Type_t::u64,     ov::element::Type_t::i64,
            ov::element::Type_t::bf16,   ov::element::Type_t::f16,     ov::element::Type_t::f32,
            ov::element::Type_t::f64,    ov::element::Type_t::boolean, ov::element::Type_t::string,
            ov::element::Type_t::nf4,    ov::element::Type_t::f4e2m1,  ov::element::Type_t::f8e8m0,
            ov::element::Type_t::dynamic};

        if (supported_precisions.find(input_precision) == supported_precisions.end()) {
            OPENVINO_THROW_NOT_IMPLEMENTED("CPU plugin: Input image format ",
                                           input_precision,
                                           " is not supported yet...");
        }
    }

    const auto& config = orig_config;
    const std::shared_ptr<ov::Model> cloned_model = model->clone();
    Config::ModelType modelType = getModelType(model);
    DEBUG_LOG(PrintableModel(*cloned_model, "org_"));

    // update the props after the perf mode translated to configs
    // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
    Config conf = engConfig;
    conf.applyRtInfo(cloned_model);
    conf.readProperties(config, modelType);

    Transformations transformations(cloned_model, conf);

    transformations.UpToLpt();

    calculate_streams(conf, cloned_model);

    if (!conf.cacheEncrypt || !conf.cacheDecrypt) {
        conf.cacheEncrypt = codec_xor_str;
        conf.cacheDecrypt = codec_xor_str;
    }

    transformations.PostLpt();
    transformations.Snippets();

    transformations.CpuSpecificOpSet();

    DEBUG_LOG(PrintableModel(*cloned_model, "cpu_"));

    OPENVINO_ASSERT(cloned_model->inputs().size() == model->inputs().size() &&
                        cloned_model->outputs().size() == model->outputs().size(),
                    "Input/output ports count mismatch between the original model and after the transformation! "
                    "Original model inputs count: ",
                    model->inputs().size(),
                    " after the transformations ",
                    cloned_model->inputs().size(),
                    ". Original model outputs count:",
                    model->inputs().size(),
                    " after the transformations ",
                    cloned_model->outputs().size());
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
        return static_cast<decltype(ov::optimal_number_of_infer_requests)::value_type>(
            streams);  // ov::optimal_number_of_infer_requests has no negative values
    }
    if (name == ov::num_streams) {
        const auto streams = engConfig.streamExecutorConfig.get_streams();
        return decltype(ov::num_streams)::value_type(
            streams);  // ov::num_streams has special negative values (AUTO = -1, NUMA = -2)
    }
    if (name == ov::device::id.name()) {
        return decltype(ov::device::id)::value_type{engConfig.device_id};
    }
    if (name == ov::inference_num_threads) {
        const auto threads = engConfig.streamExecutorConfig.get_threads();
        return static_cast<decltype(ov::inference_num_threads)::value_type>(threads);
    }
    if (name == ov::enable_profiling.name()) {
        const bool perfCount = engConfig.collectPerfCounters;
        return static_cast<decltype(ov::enable_profiling)::value_type>(perfCount);
    }
    if (name == ov::hint::inference_precision) {
        return decltype(ov::hint::inference_precision)::value_type(engConfig.inferencePrecision);
    }
    if (name == ov::hint::performance_mode) {
        return engConfig.hintPerfMode;
    }
    if (name == ov::hint::enable_cpu_pinning) {
        const bool pin_value = engConfig.enableCpuPinning;
        return static_cast<decltype(ov::hint::enable_cpu_pinning)::value_type>(pin_value);
    }
    if (name == ov::hint::enable_cpu_reservation) {
        const bool reserve_value = engConfig.enableCpuReservation;
        return static_cast<decltype(ov::hint::enable_cpu_reservation)::value_type>(reserve_value);
    }
    if (name == ov::hint::scheduling_core_type) {
        const auto core_type = engConfig.schedulingCoreType;
        return core_type;
    }
    if (name == ov::hint::model_distribution_policy) {
        const auto& distribution_policy = engConfig.modelDistributionPolicy;
        return distribution_policy;
    }
    if (name == ov::hint::enable_hyper_threading) {
        const bool ht_value = engConfig.enableHyperThreading;
        return static_cast<decltype(ov::hint::enable_hyper_threading)::value_type>(ht_value);
    }
    if (name == ov::hint::num_requests) {
        return static_cast<decltype(ov::hint::num_requests)::value_type>(engConfig.hintNumRequests);
    }
    if (name == ov::hint::execution_mode) {
        return engConfig.executionMode;
    }
    if (name == ov::internal::compiled_model_runtime_properties.name()) {
        auto model_runtime_properties = ov::Any(m_compiled_model_runtime_properties);
        return decltype(ov::internal::compiled_model_runtime_properties)::value_type(
            std::move(model_runtime_properties.as<std::string>()));
    }
    if (name == ov::log::level) {
        return engConfig.logLevel;
    }
    if (name == ov::internal::compiled_model_runtime_properties_supported.name()) {
        ov::Any res = true;
        auto it = options.find(ov::internal::compiled_model_runtime_properties.name());
        if (it == options.end()) {
            res = false;
        } else {
            ov::AnyMap input_map = it->second.as<ov::AnyMap>();
            for (const auto& item : m_compiled_model_runtime_properties) {
                auto it = input_map.find(item.first);
                if (it == input_map.end() || it->second.as<std::string>() != item.second.as<std::string>()) {
                    res = false;
                    break;
                }
            }
        }
        return res;
    }
    if (name == ov::internal::exclusive_async_requests.name()) {
        return engConfig.exclusiveAsyncRequests;
    }

    if (name == ov::hint::dynamic_quantization_group_size) {
        return static_cast<decltype(ov::hint::dynamic_quantization_group_size)::value_type>(
            engConfig.fcDynamicQuantizationGroupSize);
    }

    if (name == ov::hint::kv_cache_precision) {
        return decltype(ov::hint::kv_cache_precision)::value_type(engConfig.kvCachePrecision);
    }

    if (name == ov::key_cache_precision) {
        return decltype(ov::key_cache_precision)::value_type(engConfig.keyCachePrecision);
    }

    if (name == ov::value_cache_precision) {
        return decltype(ov::value_cache_precision)::value_type(engConfig.valueCachePrecision);
    }

    if (name == ov::key_cache_group_size) {
        return static_cast<decltype(ov::key_cache_group_size)::value_type>(engConfig.keyCacheGroupSize);
    }

    if (name == ov::value_cache_group_size) {
        return decltype(ov::value_cache_group_size)::value_type(engConfig.valueCacheGroupSize);
    }

    return get_ro_property(name, options);
}

ov::Any Plugin::get_ro_property(const std::string& name, [[maybe_unused]] const ov::AnyMap& options) const {
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
            RW_property(ov::hint::enable_cpu_reservation.name()),
            RW_property(ov::hint::scheduling_core_type.name()),
            RW_property(ov::hint::model_distribution_policy.name()),
            RW_property(ov::hint::enable_hyper_threading.name()),
            RW_property(ov::device::id.name()),
            RW_property(ov::intel_cpu::denormals_optimization.name()),
            RW_property(ov::log::level.name()),
            RW_property(ov::intel_cpu::sparse_weights_decompression_rate.name()),
            RW_property(ov::intel_cpu::enable_tensor_parallel.name()),
            RW_property(ov::hint::dynamic_quantization_group_size.name()),
            RW_property(ov::hint::kv_cache_precision.name()),
            RW_property(ov::key_cache_precision.name()),
            RW_property(ov::value_cache_precision.name()),
            RW_property(ov::key_cache_group_size.name()),
            RW_property(ov::value_cache_group_size.name()),
        };

        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(roProperties.size() + rwProperties.size());
        supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
        supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

        return decltype(ov::supported_properties)::value_type(std::move(supportedProperties));
    }

    if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
#if !defined(OPENVINO_ARCH_ARM) && !(defined(__APPLE__) || defined(__MACOSX))
            ov::PropertyName{ov::internal::caching_with_mmap.name(), ov::PropertyMutability::RO},
#endif
            ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::compiled_model_runtime_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::compiled_model_runtime_properties_supported.name(),
                             ov::PropertyMutability::RO}};
    }
    if (name == ov::device::full_name) {
        return decltype(ov::device::full_name)::value_type(deviceFullName);
    }
    if (name == ov::available_devices) {
        const std::vector<std::string> availableDevices = {""};
        return decltype(ov::available_devices)::value_type(availableDevices);
    }
    if (name == ov::device::capabilities) {
        std::vector<std::string> capabilities;
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16) ||
            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2)) {
            capabilities.emplace_back(ov::device::capability::BF16);
        }
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
            capabilities.emplace_back(ov::device::capability::WINOGRAD);
        }
        capabilities.emplace_back(ov::device::capability::FP32);
        if (hasHardwareSupport(ov::element::f16)) {
            capabilities.emplace_back(ov::device::capability::FP16);
        }
        capabilities.emplace_back(ov::device::capability::INT8);
        capabilities.emplace_back(ov::device::capability::BIN);
        capabilities.emplace_back(ov::device::capability::EXPORT_IMPORT);
        return decltype(ov::device::capabilities)::value_type(std::move(capabilities));
    }
    if (name == ov::range_for_async_infer_requests) {
        const std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 1, 1);
        return decltype(ov::range_for_async_infer_requests)::value_type(range);
    }
    if (name == ov::range_for_streams) {
        const std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, parallel_get_max_threads());
        return decltype(ov::range_for_streams)::value_type(range);
    }
    if (name == ov::internal::caching_properties) {
        std::vector<ov::PropertyName> cachingProperties = {ov::device::full_name};
        return decltype(ov::internal::caching_properties)::value_type(std::move(cachingProperties));
    }
    if (name == ov::intel_cpu::denormals_optimization) {
        return static_cast<decltype(ov::intel_cpu::denormals_optimization)::value_type>(
            engConfig.denormalsOptMode == Config::DenormalsOptMode::DO_On);
    }
    if (name == ov::intel_cpu::sparse_weights_decompression_rate) {
        return static_cast<decltype(ov::intel_cpu::sparse_weights_decompression_rate)::value_type>(
            engConfig.fcSparseWeiDecompressionRate);
    }
    if (name == ov::intel_cpu::enable_tensor_parallel) {
        return static_cast<decltype(ov::intel_cpu::enable_tensor_parallel)::value_type>(engConfig.enableTensorParallel);
    }
    if (name == ov::execution_devices) {
        return decltype(ov::execution_devices)::value_type{get_device_name()};
    }
    if (name == ov::device::type) {
        return static_cast<decltype(ov::device::type)::value_type>(ov::device::Type::INTEGRATED);
    }
    if (name == ov::device::architecture) {
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

    OPENVINO_ASSERT(model, "Only ngraph-based models are supported!");
    Config conf = engConfig;
    Config::ModelType modelType = getModelType(model);
    conf.applyRtInfo(model);
    conf.readProperties(config, modelType);

    auto context = std::make_shared<GraphContext>(conf, fake_w_cache, false);

    auto supported = ov::get_supported_nodes(
        model,
        [&](std::shared_ptr<ov::Model>& model) {
            Transformations transformation(model, conf);
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

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model_stream, const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "import_model");

    CacheDecrypt decrypt{codec_xor};
    bool decript_from_string = false;
    if (auto it = config.find(ov::cache_encryption_callbacks.name()); it != config.end()) {
        const auto& encryption_callbacks = it->second.as<EncryptionCallbacks>();
        decrypt.m_decrypt_str = encryption_callbacks.decrypt;
        decript_from_string = true;
    }

    ModelDeserializer deserializer(
        model_stream,
        [this](const std::shared_ptr<ov::AlignedBuffer>& model, const std::shared_ptr<ov::AlignedBuffer>& weights) {
            return get_core()->read_model(model, weights);
        },
        decrypt,
        decript_from_string);

    return deserialize_model(deserializer, config);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model_tensor,
                                                         const ov::AnyMap& config) const {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::intel_cpu_LT, "import_model");

    CacheDecrypt decrypt{codec_xor};
    bool decript_from_string = false;
    if (auto it = config.find(ov::cache_encryption_callbacks.name()); it != config.end()) {
        const auto& encryption_callbacks = it->second.as<EncryptionCallbacks>();
        decrypt.m_decrypt_str = encryption_callbacks.decrypt;
        decript_from_string = true;
    }

    std::shared_ptr<ov::AlignedBuffer> model_buffer =
        std::make_shared<ov::SharedBuffer<ov::Tensor>>(reinterpret_cast<char*>(model_tensor.data()),
                                                       model_tensor.get_byte_size(),
                                                       model_tensor);

    ModelDeserializer deserializer(
        model_buffer,
        [this](const std::shared_ptr<ov::AlignedBuffer>& model, const std::shared_ptr<ov::AlignedBuffer>& weights) {
            return get_core()->read_model(model, weights);
        },
        decrypt,
        decript_from_string);

    return deserialize_model(deserializer, config);
}

std::shared_ptr<ov::ICompiledModel> Plugin::deserialize_model(ModelDeserializer& deserializer,
                                                              const ov::AnyMap& config) const {
    std::shared_ptr<ov::Model> model;
    deserializer >> model;

    auto _config = config;
    Config conf = engConfig;
    Config::ModelType modelType = getModelType(model);
    conf.applyRtInfo(model);
    // check ov::loaded_from_cache property and erase it to avoid exception in readProperties.
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
}  // namespace ov::intel_cpu

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
