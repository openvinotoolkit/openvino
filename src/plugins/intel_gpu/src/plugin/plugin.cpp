// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/plugin.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/device_query.hpp"
#include "intel_gpu/runtime/dispatch_probe.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/profiling.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/plugin_config.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

#ifdef OV_GPU_MLIR_BACKEND_LINKED
#    include "transformations/mlir/interface/gpu_runtime.hpp"
#endif

// Undef DEVICE_TYPE macro which can be defined somewhere in windows headers as DWORD and conflict with our metric
#ifdef DEVICE_TYPE
#undef DEVICE_TYPE
#endif

namespace ov::intel_gpu {

std::string Plugin::get_device_id_from_config(const ov::AnyMap& config) {
    std::string id;
    if (config.find(ov::device::id.name()) != config.end()) {
        id = config.at(ov::device::id.name()).as<std::string>();
    }
    return id;
}

std::string Plugin::get_device_id(const ov::AnyMap& config) const {
    std::string id = m_default_device_id;
    if (config.find(ov::device::id.name()) != config.end()) {
        id = config.at(ov::device::id.name()).as<std::string>();
    }
    return id;
}

// weak map to hold singleton default contexts
// Weak singleton map is used to share contexts between multiple plugin(or Core) instances.
// It is needed to ensure that context is released before plugin is unloaded.
// As the actual ownership is in plugin class, the ownership is released when plugin is destructed.
std::map<std::string, std::weak_ptr<RemoteContextImpl>> weak_singleton_default_contexts;
std::mutex singleton_default_contexts_mutex;

std::map<std::string, RemoteContextImpl::Ptr> Plugin::get_default_contexts() const {
    std::call_once(m_default_contexts_once, [this]() {
        std::lock_guard<std::mutex> lock(singleton_default_contexts_mutex);
        for (const auto& device : m_device_map) {
            const auto device_name = get_device_name() + "." + device.first;

            // If already initialized, use existing one
            if (weak_singleton_default_contexts.find(device.first) != weak_singleton_default_contexts.end()) {
                if (auto ctx = weak_singleton_default_contexts[device.first].lock()) {
                    m_default_contexts[device.first] = ctx;
                    continue;
                }
            }
            // If context is not created yet or expired, create new one
            const auto initialize = false;
            auto ctx = std::make_shared<RemoteContextImpl>(device_name, std::vector<cldnn::device::ptr>{device.second}, initialize);
            weak_singleton_default_contexts[device.first] = ctx;
            m_default_contexts[device.first] = ctx;
        }
    });
    return m_default_contexts;
}

Plugin::Plugin() {
#ifdef OV_GPU_MLIR_BACKEND_LINKED
    ov::intel_gpu::mlir::register_mlir_gpu_runtime();
#endif
    set_device_name("GPU");
    register_primitives();

    cldnn::device_query device_query;
    m_device_map = device_query.get_available_devices();

    // Set default configs for each device
    for (const auto& device : m_device_map) {
        m_configs_map.insert({device.first, ExecutionConfig(ov::device::id(device.first))});
    }

    // Set common info for compiled_model_runtime_properties
    const auto& ov_version = ov::get_openvino_version();
    m_compiled_model_runtime_properties["OV_VERSION"] = ov_version.buildNumber;
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& remote_properties) const {
    if (remote_properties.empty()) {
        return get_default_context(m_default_device_id);
    }
    return std::make_shared<RemoteContextImpl>(get_default_contexts(), remote_properties);
}

std::shared_ptr<RemoteContextImpl> Plugin::get_default_context(const std::string& device_id, bool initialize) const {
    auto contexts = get_default_contexts();
    OPENVINO_ASSERT(contexts.contains(device_id), "[GPU] Context was not initialized for ", device_id, " device");

    auto context = contexts.at(device_id);
    if (initialize) {
        context->initialize();
    }

    return context;
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const AnyMap& params) const {
    std::string device_id = m_default_device_id;

    if (params.find(ov::device::id.name()) != params.end()) {
        device_id = params.at(ov::device::id.name()).as<std::string>();
    }

    return get_default_context(device_id);
}

void Plugin::apply_device_id_map(const std::map<std::string, std::string>& id_map) {
    // Core owns the numbering when several libraries share a device name; adopt its ids for every
    // device we enumerated, so nothing downstream (contexts, configs, metrics) sees our own.
    std::map<std::string, cldnn::device::ptr> renamed_devices;
    std::map<std::string, ExecutionConfig> renamed_configs;
    for (const auto& [internal_id, device] : m_device_map) {
        const auto it = id_map.find(internal_id);
        OPENVINO_ASSERT(it != id_map.end(), "[GPU] Core did not assign an id for enumerated device \"", internal_id, "\", so it could not be addressed.");
        renamed_devices[it->second] = device;
        renamed_configs.emplace(it->second, ExecutionConfig(ov::device::id(it->second)));
    }
    m_device_map = std::move(renamed_devices);
    m_configs_map = std::move(renamed_configs);
    // The assigned ids need not contain "0": another member may hold the device Core numbered 0.
    m_default_device_id = m_device_map.empty() ? std::string("0") : m_device_map.begin()->first;
}

void Plugin::set_property(const ov::AnyMap &config) {
    for (const auto& [name, value] : config) {
        OPENVINO_ASSERT(supports_runtime_property(name), "[GPU] Property ", name, " is not in a list of supported properties");
    }

    auto update_config = [](ExecutionConfig& config, const ov::AnyMap& user_config) {
        config.set_user_property(user_config, OptionVisibility::RELEASE);
        validate_custom_layer_config(user_config);
    };

    auto rest = config;
    if (auto id_map = rest.extract(ov::internal::device_id_map.name()); !id_map.empty()) {
        apply_device_id_map(id_map.mapped().as<std::map<std::string, std::string>>());
        if (!rest.empty()) {
            set_property(rest);
        }
        return;
    }

    if (config.find(ov::internal::config_device_id.name()) != config.end()) {
        std::string device_id = config.at(ov::internal::config_device_id.name()).as<std::string>();
        auto config_for_device = config;
        config_for_device.erase(ov::internal::config_device_id.name());
        update_config(m_configs_map.at(device_id), config_for_device);
    } else {
        std::string device_id = get_device_id_from_config(config);
        if (!device_id.empty()) {
            m_default_device_id = device_id;
            update_config(m_configs_map.at(device_id), config);
        } else {
            for (auto& conf : m_configs_map) {
                update_config(conf.second, config);
            }
        }
    }
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& options) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::get_property");
    OPENVINO_ASSERT(supports_runtime_property(name), "[GPU] Property ", name, " is not in a list of supported properties");

    // The metrics below don't depend on the device ID, so we should handle those
    // earler than querying actual ID to avoid exceptions when no devices are found
    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {get_supported_properties()};
    }
    if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{get_supported_internal_properties()};
    }
    if (name == ov::available_devices) {
        std::vector<std::string> available_devices = { };
        for (const auto& dev : m_device_map) {
            available_devices.push_back(dev.first);
        }
        return decltype(ov::available_devices)::value_type {available_devices};
    }
    if (name == ov::internal::caching_properties) {
        return decltype(ov::internal::caching_properties)::value_type(get_caching_properties());
    }

    ov::AnyMap actual_runtime_info;
    auto prepare_actual_runtime_info = [&]() {
        // Items can be inserted here if needed.
        // Note: driver version used to be here, but it was moved to cache load logic. See
        // https://github.com/openvinotoolkit/openvino/pull/29195
    };
    // Below properties depend on the device ID.
    if (name == ov::internal::compiled_model_runtime_properties.name()) {
        prepare_actual_runtime_info();
        auto model_runtime_info = m_compiled_model_runtime_properties;
        // Set specified device info for compiled_model_runtime_properties
        model_runtime_info.insert(actual_runtime_info.begin(), actual_runtime_info.end());
        auto model_format = ov::Any(model_runtime_info);
        return decltype(ov::internal::compiled_model_runtime_properties)::value_type(
            std::move(model_format.as<std::string>()));
    }
    if (name == ov::internal::compiled_model_runtime_properties_supported.name()) {
        ov::Any res = true;
        prepare_actual_runtime_info();
        auto it = options.find(ov::internal::compiled_model_runtime_properties.name());
        if (it == options.end()) {
            res = false;
            return res;
        }
        ov::AnyMap input_map = it->second.as<ov::AnyMap>();
        // Check common info of compiled_model_runtime_properties
        for (const auto& item : m_compiled_model_runtime_properties) {
            auto it = input_map.find(item.first);
            if (it == input_map.end() || it->second.as<std::string>() != item.second.as<std::string>()) {
                res = false;
                return res;
            }
        }
        // Check specified device info of compiled_model_runtime_properties
        for (const auto& it : actual_runtime_info) {
            auto item = input_map.find(it.first);
            if (item == input_map.end() || item->second.as<std::string>() != it.second.as<std::string>()) {
                res = false;
                break;
            }
        }
        return res;
    }

    OPENVINO_ASSERT(!m_device_map.empty(), "[GPU] Can't get ", name, " property as no supported devices found or an error happened during devices query.\n"
                                           "[GPU] Please check OpenVINO documentation for GPU drivers setup guide.\n");

    if (is_metric(name)) {
        return get_metric(name, options);
    }

    std::string device_id = m_default_device_id;
    if (options.find(ov::device::id.name()) != options.end()) {
        device_id = options.find(ov::device::id.name())->second.as<std::string>();
    }
    OPENVINO_ASSERT(m_configs_map.find(device_id) != m_configs_map.end(), "[GPU] get_property: Couldn't find config for GPU with id ", device_id);

    const auto& c = m_configs_map.at(device_id);
    return c.get_property(name, OptionVisibility::RELEASE);
}

auto StringRightTrim = [](std::string string, std::string substring, bool case_sensitive = true) {
    auto ret_str = string;
    if (!case_sensitive) {
        std::transform(string.begin(), string.end(), string.begin(), ::tolower);
        std::transform(substring.begin(), substring.end(), substring.begin(), ::tolower);
    }
    auto erase_position = string.rfind(substring);
    if (erase_position != std::string::npos) {
        // if space exists before substring remove it also
        if (std::isspace(string.at(erase_position - 1))) {
            erase_position--;
        }
        return ret_str.substr(0, erase_position);
    }
    return ret_str;
};

bool Plugin::is_metric(const std::string& name) {
    auto all_properties = get_supported_properties();
    auto internal_properties = get_supported_internal_properties();
    auto caching_properties = get_caching_properties();
    all_properties.insert(all_properties.end(), internal_properties.begin(), internal_properties.end());
    all_properties.insert(all_properties.end(), caching_properties.begin(), caching_properties.end());
    auto it = std::find(all_properties.begin(), all_properties.end(), name);
    OPENVINO_ASSERT(it != all_properties.end(), "[GPU] Property ", name, " is not in a list of supported properties");

    return !it->is_mutable();
}

ov::Any Plugin::get_metric(const std::string& name, const ov::AnyMap& options) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::get_metric");
    auto device_id = get_property(ov::device::id.name(), options).as<std::string>();

    auto iter = m_device_map.find(std::to_string(cldnn::device_query::device_id));
    if (iter == m_device_map.end()) {
        iter = m_device_map.find(device_id);
    }
    if (iter == m_device_map.end()) {
        iter = m_device_map.begin();
    }
    auto device = iter->second;
    auto device_info = device->get_info();

    if (name == ov::intel_gpu::device_total_mem_size) {
        return decltype(ov::intel_gpu::device_total_mem_size)::value_type {device_info.max_global_mem_size};
    }
    if (name == ov::intel_gpu::device_max_alloc_mem_size) {
        return decltype(ov::intel_gpu::device_max_alloc_mem_size)::value_type {device_info.max_alloc_mem_size};
    }
    if (name == ov::device::type) {
        auto dev_type = device_info.dev_type == cldnn::device_type::discrete_gpu ? ov::device::Type::DISCRETE : ov::device::Type::INTEGRATED;
        return decltype(ov::device::type)::value_type {dev_type};
    }
    if (name == ov::device::gops) {
        std::map<element::Type, float> gops;
        gops[element::i8] = device->get_gops(cldnn::data_types::i8);
        gops[element::u8] = device->get_gops(cldnn::data_types::u8);
        gops[element::f16] = device->get_gops(cldnn::data_types::f16);
        gops[element::f32] = device->get_gops(cldnn::data_types::f32);
        return decltype(ov::device::gops)::value_type {gops};
    }
    if (name == ov::intel_gpu::cacheline_size) {
        return static_cast<decltype(ov::intel_gpu::cacheline_size)::value_type>(device_info.cacheline_size.value_or(0));
    }
    if (name == ov::intel_gpu::execution_units_count) {
        return static_cast<decltype(ov::intel_gpu::execution_units_count)::value_type>(device_info.execution_units_count);
    }
    if (name == ov::intel_gpu::uarch_version) {
        std::stringstream s;
        if (device_info.gfx_ver.major == 0 && device_info.gfx_ver.minor == 0 && device_info.gfx_ver.revision == 0) {
            s << "unknown";
        } else {
            s << static_cast<int>(device_info.gfx_ver.major) << "."
              << static_cast<int>(device_info.gfx_ver.minor) << "."
              << static_cast<int>(device_info.gfx_ver.revision);
        }
        return decltype(ov::intel_gpu::uarch_version)::value_type {s.str()};
    }
    if (name == ov::device::uuid) {
        return decltype(ov::device::uuid)::value_type {device_info.uuid};
    }
    if (name == ov::device::luid) {
        return decltype(ov::device::luid)::value_type {device_info.luid};
    }
    if (name == ov::device::full_name) {
        auto deviceName = StringRightTrim(device_info.dev_name, "NEO", false);
        deviceName += std::string(" (") + (device_info.dev_type == cldnn::device_type::discrete_gpu ? "dGPU" : "iGPU") + ")";
        return decltype(ov::device::full_name)::value_type {deviceName};
    }
    if (name == ov::device::capabilities) {
        return supports_model_execution() ? get_device_capabilities(device_info) : std::vector<std::string>{};
    }
    if (name == ov::range_for_streams) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, device_info.num_ccs == 1 ? 2 : device_info.num_ccs);
        return decltype(ov::range_for_streams)::value_type {range};
    }
    if (name == ov::intel_gpu::memory_statistics) {
        const auto& ctx = get_default_context(device_id, false);
        if (!ctx->is_initialized()) {
            decltype(ov::intel_gpu::memory_statistics)::value_type statistics;
            for (const auto type : ctx->get_device().get_mem_caps().get_memory_statistics_types()) {
                std::ostringstream name;
                name << type;
                statistics[name.str()] = 0;
            }
            return statistics;
        }
        return decltype(ov::intel_gpu::memory_statistics)::value_type{ctx->get_engine().get_memory_statistics()};
    }
    if (name == ov::intel_gpu::runtime_type) {
        return decltype(ov::intel_gpu::runtime_type)::value_type {std::string(cldnn::get_runtime_cache_tag())};
    }
    if (name == ov::intel_gpu::driver_version) {
        return decltype(ov::intel_gpu::driver_version)::value_type {device_info.driver_version};
    }
    if (name == ov::intel_gpu::device_id) {
        std::stringstream s;
        s << "0x" << std::hex << device_info.device_id;
        return decltype(ov::intel_gpu::device_id)::value_type {s.str()};
    }
    if (name == ov::device::architecture) {
        std::stringstream s;
        s << "GPU: vendor=0x" << std::hex << device_info.vendor_id << std::dec << " arch=";
        if (device_info.gfx_ver.major == 0 && device_info.gfx_ver.minor == 0) {
            s << device_info.dev_name;
        } else {
            s << "v" << static_cast<int>(device_info.gfx_ver.major)
              << "." << static_cast<int>(device_info.gfx_ver.minor)
              << "." << static_cast<int>(device_info.gfx_ver.revision);
        }
        return decltype(ov::device::architecture)::value_type {s.str()};
    }
    if (name == ov::device::pci_info) {
        ov::device::PCIInfo info;
        info.domain = device_info.pci_info.pci_domain;
        info.bus = device_info.pci_info.pci_bus;
        info.device = device_info.pci_info.pci_device;
        info.function = device_info.pci_info.pci_function;
        return decltype(ov::device::pci_info)::value_type {info};
    }
    if (name == ov::internal::cache_header_alignment) {
        return decltype(ov::internal::cache_header_alignment)::value_type{
            static_cast<decltype(ov::internal::cache_header_alignment)::value_type>(ov::util::get_system_page_size())};
    }
    if (auto metric = get_model_metric(name, options, device_info); !metric.empty()) {
        return metric;
    }
    OPENVINO_THROW("Unsupported metric key ", name);
}

std::vector<ov::PropertyName> Plugin::filter_runtime_properties(std::vector<ov::PropertyName> properties) {
    const auto unsupported = [](const ov::PropertyName& property) {
        return !supports_runtime_property(property);
    };
    properties.erase(std::remove_if(properties.begin(), properties.end(), unsupported), properties.end());
    return properties;
}

std::vector<ov::PropertyName> Plugin::get_caching_properties() {
    static const auto caching_properties = filter_runtime_properties({
        ov::PropertyName{ov::device::architecture.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::execution_units_count.name(), PropertyMutability::RO},
        ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::dynamic_quantization_group_size.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::activations_scale_factor.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::kv_cache_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::runtime_type.name(), PropertyMutability::RO},
    });

    return caching_properties;
}

std::vector<ov::PropertyName> Plugin::get_supported_properties() {
    static const auto supported_properties = filter_runtime_properties({
        // Metrics
        ov::PropertyName{ov::supported_properties.name(), PropertyMutability::RO},
        ov::PropertyName{ov::available_devices.name(), PropertyMutability::RO},
        ov::PropertyName{ov::range_for_async_infer_requests.name(), PropertyMutability::RO},
        ov::PropertyName{ov::range_for_streams.name(), PropertyMutability::RO},
        ov::PropertyName{ov::optimal_batch_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::max_batch_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::architecture.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::full_name.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::uuid.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::luid.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::type.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::gops.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::capabilities.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::pci_info.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::device_id.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::device_total_mem_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::device_max_alloc_mem_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::uarch_version.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::cacheline_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::execution_units_count.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::memory_statistics.name(), PropertyMutability::RO},
        ov::PropertyName{ov::compatibility_check.name(), PropertyMutability::RO},

        // Configs
        ov::PropertyName{ov::enable_profiling.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::model_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::host_task_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::queue_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::queue_throttle.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::enable_sdpa_optimization.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::enable_lora_operation.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::enable_large_allocations.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::enable_loop_unrolling.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::disable_winograd_convolution.name(), PropertyMutability::RW},
        ov::PropertyName{ov::cache_dir.name(), PropertyMutability::RW},
        ov::PropertyName{ov::cache_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::compilation_num_threads.name(), PropertyMutability::RW},
        ov::PropertyName{ov::num_streams.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::num_requests.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::enable_cpu_pinning.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::enable_cpu_reservation.name(), PropertyMutability::RW},
        ov::PropertyName{ov::device::id.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::dynamic_quantization_group_size.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::activations_scale_factor.name(), PropertyMutability::RW},
        ov::PropertyName{ov::weights_path.name(), PropertyMutability::RW},
        ov::PropertyName{ov::cache_encryption_callbacks.name(), PropertyMutability::WO},
        ov::PropertyName{ov::hint::kv_cache_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::model.name(), PropertyMutability::WO},
        ov::PropertyName{ov::intel_gpu::offload_ratio.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::config_file.name(), PropertyMutability::RW},
    });
    return supported_properties;
}

std::vector<ov::PropertyName> Plugin::get_supported_internal_properties() {
    static const auto supported_internal_properties = filter_runtime_properties({
        ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::internal::config_device_id.name(), ov::PropertyMutability::WO},
        ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::internal::compiled_model_runtime_properties.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::internal::compiled_model_runtime_properties_supported.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::internal::query_model_ratio.name(), PropertyMutability::RW},
        ov::PropertyName{ov::internal::caching_with_mmap.name(), PropertyMutability::RO},
        ov::PropertyName{ov::internal::cache_header_alignment.name(), PropertyMutability::RO},
        ov::PropertyName{ov::internal::device_id_map.name(), ov::PropertyMutability::WO},
    });
    return supported_internal_properties;
}

std::vector<std::string> Plugin::get_device_capabilities(const cldnn::device_info& info) {
    std::vector<std::string> capabilities;

    capabilities.emplace_back(ov::device::capability::FP32);
    capabilities.emplace_back(ov::device::capability::BIN);
    if (info.supports_fp16) {
        capabilities.emplace_back(ov::device::capability::FP16);
    }
    if (info.supports_bf16) {
        capabilities.emplace_back(ov::device::capability::BF16);
    }
    if (info.supports_imad || info.supports_immad) {
        capabilities.emplace_back(ov::device::capability::INT8);
    }
    if (info.supports_immad) {
        capabilities.emplace_back(ov::intel_gpu::capability::HW_MATMUL);
    }
    if (info.supports_usm) {
        capabilities.emplace_back(ov::intel_gpu::capability::USM_MEMORY);
    }
    capabilities.emplace_back(ov::device::capability::EXPORT_IMPORT);

    return capabilities;
}

}  // namespace ov::intel_gpu

namespace {

void enumerate_dispatch_devices(std::vector<ov::EnumeratedDevice>& out) {
    out.clear();  // The probe reports, it does not append: repeated calls must be idempotent.
    try {
        // Read once per process: the debug override cannot change while the library is loaded.
        static const std::string forced = ov::util::getenv_string("OV_GPU_RUNTIME");

        for (const auto& d : cldnn::lightweight_enumerate()) {
            ov::EnumeratedDevice e;
            e.internal_id = d.map_id;
            e.fingerprint = cldnn::make_fingerprint(d.info);
            e.score = cldnn::probe_score(cldnn::get_default_runtime_type(), d.info, forced);
            out.push_back(std::move(e));
        }
    } catch (...) {
        out.clear();  // Any failure -> "I serve nothing"; the probe never throws.
    }
}

}  // namespace

static const ov::Version version = { CI_BUILD_NUMBER, "Intel GPU plugin" };
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::intel_gpu::Plugin, version)

// Real implementation of the plugin-ABI enumeration probe (the GPU plugin participates
// in device-name dispatch); mirrors the stub every other plugin exports.
OV_DEFINE_PLUGIN_ENUMERATE_FUNCTION(enumerate_dispatch_devices)
