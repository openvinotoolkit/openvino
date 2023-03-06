// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <tuple>
#include <cctype>
#include <memory>
#include "ie_metric_helpers.hpp"
#include <ie_ngraph_utils.hpp>
#include <ie_algorithm.hpp>

#include "openvino/runtime/intel_gpu/properties.hpp"
#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/plugin/legacy_api_helper.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/device_query.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "ie_plugin_config.hpp"
#include "gpu/gpu_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_icore.hpp"

#include "dimension_tracker.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/utils/utils.hpp>

#include <openvino/pass/manager.hpp>
#include <openvino/util/common_util.hpp>

#include <performance_heuristics.hpp>

// Undef DEVICE_TYPE macro which can be defined somewhere in windows headers as DWORD and conflict with our metric
#ifdef DEVICE_TYPE
#undef DEVICE_TYPE
#endif

using namespace InferenceEngine;
using namespace InferenceEngine::gpu;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_gpu {

#define FACTORY_DECLARATION(op_version, op_name) \
    void __register ## _ ## op_name ## _ ## op_version();

#define FACTORY_CALL(op_version, op_name) \
    __register ## _ ## op_name ## _ ## op_version();

#define REGISTER_FACTORY(op_version, op_name) FACTORY_DECLARATION(op_version, op_name)
#include "intel_gpu/plugin/primitives_list.hpp"
#undef REGISTER_FACTORY

void Plugin::register_primitives() {
    #define REGISTER_FACTORY(op_version, op_name) FACTORY_CALL(op_version, op_name)
    #include "intel_gpu/plugin/primitives_list.hpp"
    #undef REGISTER_FACTORY
}

ov::AnyMap Plugin::preprocess_config(const std::map<std::string, std::string>& orig_config) const {
    // We can skip this conversion for new API once all meta plugins don't try to use legacy configs/metrics for new API internally
    auto config = LegacyAPIHelper::convert_legacy_properties(orig_config, IsNewAPI());

    // Code below is WA for issue 100498
    auto hint_it = std::find_if(orig_config.begin(), orig_config.end(), [](const std::pair<std::string, ov::Any>& kv) {
        return kv.first == ov::hint::performance_mode.name();
    });

    if (hint_it != orig_config.end()) {
        config[ov::hint::performance_mode.name()] = ov::util::from_string(hint_it->second, ov::hint::performance_mode);
    }

    return config;
}

std::string Plugin::get_device_id_from_config(const std::map<std::string, std::string>& config) const {
    std::string device_id;
    if (config.find(PluginConfigParams::KEY_DEVICE_ID) != config.end()) {
        device_id = config.at(PluginConfigParams::KEY_DEVICE_ID);
    }
    return device_id;
}

std::string Plugin::get_device_id(const std::map<std::string, std::string>& config) const {
    std::string device_id = default_device_id;
    if (config.find(PluginConfigParams::KEY_DEVICE_ID) != config.end()) {
        device_id = config.at(PluginConfigParams::KEY_DEVICE_ID);
    }
    return device_id;
}

void Plugin::transform_model(std::shared_ptr<ov::Model>& model, const ExecutionConfig& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::transform_model");
    auto deviceInfo = device_map.at(config.get_property(ov::device::id))->get_info();
    TransformationsPipeline transformations(config, deviceInfo);
    transformations.apply(model);
}

InferenceEngine::CNNNetwork Plugin::clone_and_transform_model(const InferenceEngine::CNNNetwork& network,
                                                             const ExecutionConfig& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::clone_and_transform_model");
    GPU_DEBUG_DEFINE_MEM_LOGGER("Plugin::clone_and_transform_model");
    CNNNetwork clonedNetwork = InferenceEngine::details::cloneNetwork(network);

    auto nGraphFunc = clonedNetwork.getFunction();
    if (nGraphFunc) {
        transform_model(nGraphFunc, config);
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(!debug_config->dump_graphs.empty()) {
            auto path_base = debug_config->dump_graphs + "/" + network.getName() + "_" +  "transformed_func";
            ov::pass::Serialize(path_base + ".xml", path_base + ".bin").run_on_model(nGraphFunc);
    }
    }
    return clonedNetwork;
}

Plugin::Plugin() : m_default_contexts({}) {
    _pluginName = "GPU";
    register_primitives();
    // try loading gpu engine and get info from it
    {
        // Set OCL runtime which should be always available
        cldnn::device_query device_query(cldnn::engine_types::ocl, cldnn::runtime_types::ocl);
        device_map = device_query.get_available_devices();

        // Set default configs for each device
        for (auto& device : device_map) {
            m_configs_map.insert({device.first, ExecutionConfig(ov::device::id(device.first))});
            auto ctx = std::make_shared<RemoteCLContext>(GetName() + "." + device.first, std::vector<cldnn::device::ptr>{ device.second });
            m_default_contexts.insert({device.first, ctx});
        }
    }
}

auto check_inputs = [](InferenceEngine::InputsDataMap _networkInputs) {
    for (auto ii : _networkInputs) {
        auto input_precision = ii.second->getTensorDesc().getPrecision();
        if (input_precision != InferenceEngine::Precision::FP16 &&
            input_precision != InferenceEngine::Precision::FP32 &&
            input_precision != InferenceEngine::Precision::FP64 &&
            input_precision != InferenceEngine::Precision::U8 &&
            input_precision != InferenceEngine::Precision::I8 &&
            input_precision != InferenceEngine::Precision::I16 &&
            input_precision != InferenceEngine::Precision::U16 &&
            input_precision != InferenceEngine::Precision::I32 &&
            input_precision != InferenceEngine::Precision::U32 &&
            input_precision != InferenceEngine::Precision::I64 &&
            input_precision != InferenceEngine::Precision::U64 &&
            input_precision != InferenceEngine::Precision::BOOL) {
            IE_THROW(NotImplemented)
                << "Input image format " << input_precision << " is not supported yet...";
        }
    }
};

void Plugin::update_memory_statistics(const RemoteContextImpl::Ptr& context) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::update_memory_statistics");
    {
        std::lock_guard<std::mutex> lock(engine_mutex);

        // if the same context exists, the statistics is replaced with the latest one
        // (currently, memory usage is accumulated for several networks in the same context)
        // if it does not exist, a new statistics is added
        statistics_map[context] = context->get_engine().get_memory_statistics();
    }
}

IExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                           const std::map<std::string, std::string> &orig_config) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::LoadExeNetworkImpl");
    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs = network.getInputsInfo();
    check_inputs(_networkInputs);

    std::string device_id = get_device_id(orig_config);

    auto context = get_default_context(device_id);

    OPENVINO_ASSERT(m_configs_map.find(device_id) != m_configs_map.end(), "[GPU] LoadExeNetworkImpl: Couldn't find config for GPU with id ", device_id);

    ExecutionConfig config = m_configs_map.at(device_id);
    config.set_user_property(preprocess_config(orig_config));
    config.apply_user_properties(context->get_impl()->get_engine().get_device_info());

    auto transformedNetwork = clone_and_transform_model(network, config);
    {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::LoadExeNetworkImpl::CreateExeNetwork");
        CompiledModel::Ptr exeNetwork = std::make_shared<CompiledModel>(transformedNetwork, context, config);
        if (exeNetwork->m_graphs[0]->GetNetwork()->is_dynamic()) {
            isModelCachingEnabled = false;
        }
        update_memory_statistics(context->get_impl());
        return exeNetwork;
    }
}

IExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                           const InferenceEngine::RemoteContext::Ptr &context,
                                                           const std::map<std::string, std::string> &orig_config) {
    InferenceEngine::InputsDataMap _networkInputs = network.getInputsInfo();
    check_inputs(_networkInputs);

    auto context_impl = get_context_impl(context);
    auto device_id = InferenceEngine::DeviceIDParser{context_impl->get_device_name()}.getDeviceID();

    OPENVINO_ASSERT(m_configs_map.find(device_id) != m_configs_map.end(), "[GPU] LoadExeNetworkImpl: Couldn't find config for GPU with id ", device_id);

    ExecutionConfig config = m_configs_map.at(device_id);
    config.set_user_property(preprocess_config(orig_config));
    config.apply_user_properties(context_impl->get_engine().get_device_info());

    auto transformedNetwork = clone_and_transform_model(network, config);
    return std::make_shared<CompiledModel>(transformedNetwork, context, config);
}

InferenceEngine::RemoteContext::Ptr Plugin::CreateContext(const AnyMap& params) {
    if (params.empty()) {
        return get_default_context(default_device_id);
    }

    std::vector<RemoteContextImpl::Ptr> known_contexts;
    for (auto& c : m_default_contexts) {
        known_contexts.push_back(c.second->get_impl());
    }
    std::string context_type = extract_object<std::string>(params, GPU_PARAM_KEY(CONTEXT_TYPE));

    if (GPU_PARAM_VALUE(OCL) == context_type) {
        return std::make_shared<RemoteCLContext>(known_contexts, params);
    } else if (GPU_PARAM_VALUE(VA_SHARED) == context_type) {
#ifdef _WIN32
        return std::make_shared<RemoteD3DContext>(known_contexts, params);
#else
        return std::make_shared<RemoteVAContext>(known_contexts, params);
#endif
    }

    OPENVINO_ASSERT(false, "[GPU] Unsupported context type passed to CreateContext method: ", context_type);
}

RemoteCLContext::Ptr Plugin::get_default_context(const std::string& device_id) const {
    OPENVINO_ASSERT(m_default_contexts.find(device_id) != m_default_contexts.end(), "[GPU] Context was not initialized for ", device_id, " device");

    return m_default_contexts.at(device_id);;
}

InferenceEngine::RemoteContext::Ptr Plugin::GetDefaultContext(const AnyMap& params) {
    std::string device_id = default_device_id;

    if (params.find(CONFIG_KEY(DEVICE_ID)) != params.end())
        device_id = params.at(CONFIG_KEY(DEVICE_ID)).as<std::string>();

    return get_default_context(device_id);
}

void Plugin::SetConfig(const std::map<std::string, std::string> &config) {
    auto update_config = [this](ExecutionConfig& config, const std::map<std::string, std::string>& user_config) {
        config.set_user_property(preprocess_config(user_config));
        // Check that custom layers config can be loaded
        if (user_config.find(ov::intel_gpu::config_file.name()) != user_config.end()) {
            CustomLayerMap custom_layers;
            auto custom_layers_config = user_config.at(ov::intel_gpu::config_file.name());
            CustomLayer::LoadFromFile(custom_layers_config, custom_layers, custom_layers_config.empty());
        }
    };

    if (config.find(PluginConfigInternalParams::KEY_CONFIG_DEVICE_ID) != config.end()) {
        std::string device_id = config.at(PluginConfigInternalParams::KEY_CONFIG_DEVICE_ID);
        update_config(m_configs_map.at(device_id), config);
    } else {
        std::string device_id = get_device_id_from_config(config);
        if (!device_id.empty()) {
            default_device_id = device_id;
            update_config(m_configs_map.at(device_id), config);
        } else {
            for (auto& conf : m_configs_map) {
                update_config(conf.second, config);
            }
        }
    }
}

QueryNetworkResult Plugin::QueryNetwork(const CNNNetwork& network,
                                        const std::map<std::string, std::string>& orig_config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::QueryNetwork");
    QueryNetworkResult res;
    std::string device_id = get_device_id(orig_config);

    auto ctx = get_default_context(device_id)->get_impl();

    ExecutionConfig config = m_configs_map.at(device_id);
    config.set_user_property(preprocess_config(orig_config));
    config.apply_user_properties(ctx->get_engine().get_device_info());

    Program prog(ctx->get_engine(), config);
    bool dyn_shape_batch_found = false;

    auto model = network.getFunction();
    if (model == nullptr) {
        IE_THROW() << "Only ngraph-based models are supported!";
    }

    auto supported = GetSupportedNodes(model,
    [&](std::shared_ptr<ov::Model>& model) {
        std::map<std::string, ngraph::PartialShape> shapes;
        std::map<std::string, std::pair<int64_t, int64_t>> batch_dim;
        dyn_shape_batch_found = prog.IsDynBatchModel(model, shapes, batch_dim);
        transform_model(model, config);
    },
    [&](std::shared_ptr<ngraph::Node> node) {
            if (node->is_dynamic()) {
                if (!dyn_shape_batch_found)
                    return false;

                auto pshape = node->get_output_partial_shape(0);
                if (pshape.rank().is_dynamic())
                    return false;

                int dynCount = 0;
                int64_t batch_idx = -1;
                for (size_t i = 0; i < pshape.size(); i++) {
                    if (pshape[i].is_dynamic()) {
                        dynCount++;
                        if (batch_idx < 0) {
                            batch_idx = i;
                        }
                    }
                }

                if (dynCount != 1)
                    return false;  // more than one dimension is dynamic

                int64_t max_batch = pshape[batch_idx].get_max_length();
                if (max_batch <= 1)
                    return false;

                return true;
            }
            return prog.IsOpSupported(network, node);
    });

    for (auto&& layerName : supported) {
        res.supportedLayersMap.emplace(layerName, ctx->get_device_name());
    }

    return res;
}

InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::ImportNetwork(std::istream& networkModel,
                                                                       const std::map<std::string, std::string>& orig_config) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::ImportNetwork");
    std::string device_id = get_device_id(orig_config);
    auto context = get_default_context(device_id);

    ExecutionConfig config = m_configs_map.at(device_id);
    config.set_user_property(preprocess_config(orig_config));
    config.apply_user_properties(context->get_impl()->get_engine().get_device_info());

    {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::ImportNetwork::CreateExeNetwork");
        CompiledModel::Ptr exeNetwork = std::make_shared<CompiledModel>(networkModel, context, config);
        exeNetwork->SetPointerToPlugin(shared_from_this());
        update_memory_statistics(context->get_impl());
        return exeNetwork;
    }
}

Parameter Plugin::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::GetConfig");

    std::string device_id = default_device_id;
    if (options.find(ov::device::id.name()) != options.end()) {
        device_id = options.find(ov::device::id.name())->second.as<std::string>();
    }

    OPENVINO_ASSERT(m_configs_map.find(device_id) != m_configs_map.end(), "[GPU] GetConfig: Couldn't find config for GPU with id ", device_id);

    const auto& c = m_configs_map.at(device_id);
    auto actual_name = name;
    if (LegacyAPIHelper::is_legacy_property({name, nullptr}, IsNewAPI())) {
        actual_name = LegacyAPIHelper::convert_legacy_property({name, nullptr}).first;
    }

    auto val = c.get_property(actual_name);
    if (LegacyAPIHelper::is_legacy_property({name, nullptr}, IsNewAPI())) {
        val = LegacyAPIHelper::convert_to_legacy_property({actual_name, val}).second;
    }

    return val;
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

Parameter Plugin::GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::GetMetric");
    GPU_DEBUG_GET_INSTANCE(debug_config);
    auto device_id = GetConfig(ov::device::id.name(), options).as<std::string>();

    auto iter = device_map.find(std::to_string(cldnn::device_query::device_id));
    if (iter == device_map.end())
        iter = device_map.find(device_id);
    if (iter == device_map.end())
        iter = device_map.begin();
    auto device = iter->second;
    auto device_info = device->get_info();
    bool is_new_api = IsNewAPI();

    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {get_supported_properties()};
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, LegacyAPIHelper::get_supported_metrics(isModelCachingEnabled));
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { };
        for (auto const& dev : device_map)
            availableDevices.push_back(dev.first);
        return decltype(ov::available_devices)::value_type {availableDevices};
    } else if (name == ov::intel_gpu::device_total_mem_size) {
        return decltype(ov::intel_gpu::device_total_mem_size)::value_type {device_info.max_global_mem_size};
    } else if (name == ov::device::type) {
        if (is_new_api) {
            auto dev_type = device_info.dev_type == cldnn::device_type::discrete_gpu ? ov::device::Type::DISCRETE : ov::device::Type::INTEGRATED;
            return decltype(ov::device::type)::value_type {dev_type};
        } else {
            auto dev_type = device_info.dev_type == cldnn::device_type::discrete_gpu ? Metrics::DeviceType::discrete : Metrics::DeviceType::integrated;
            IE_SET_METRIC_RETURN(DEVICE_TYPE, dev_type);
        }
    } else if (name == ov::device::gops) {
        if (is_new_api) {
            std::map<element::Type, float> gops;
            gops[element::i8] = device->get_gops(cldnn::data_types::i8);
            gops[element::u8] = device->get_gops(cldnn::data_types::u8);
            gops[element::f16] = device->get_gops(cldnn::data_types::f16);
            gops[element::f32] = device->get_gops(cldnn::data_types::f32);
            return decltype(ov::device::gops)::value_type {gops};
        } else {
            std::map<InferenceEngine::Precision, float> gops;
            gops[InferenceEngine::Precision::I8] = device->get_gops(cldnn::data_types::i8);
            gops[InferenceEngine::Precision::U8] = device->get_gops(cldnn::data_types::u8);
            gops[InferenceEngine::Precision::FP16] = device->get_gops(cldnn::data_types::f16);
            gops[InferenceEngine::Precision::FP32] = device->get_gops(cldnn::data_types::f32);
            IE_SET_METRIC_RETURN(DEVICE_GOPS, gops);
        }
    } else if (name == ov::intel_gpu::execution_units_count) {
        return static_cast<decltype(ov::intel_gpu::execution_units_count)::value_type>(device_info.execution_units_count);
    } else if (name == ov::intel_gpu::uarch_version) {
        std::stringstream s;
        if (device_info.gfx_ver.major == 0 && device_info.gfx_ver.minor == 0 && device_info.gfx_ver.revision == 0) {
            s << "unknown";
        } else {
            s << static_cast<int>(device_info.gfx_ver.major) << "."
              << static_cast<int>(device_info.gfx_ver.minor) << "."
              << static_cast<int>(device_info.gfx_ver.revision);
        }
        return decltype(ov::intel_gpu::uarch_version)::value_type {s.str()};
    } else if (name == METRIC_KEY(OPTIMAL_BATCH_SIZE) ||
               name == ov::optimal_batch_size) {
        return decltype(ov::optimal_batch_size)::value_type {get_optimal_batch_size(options)};
    } else if (name == ov::device::uuid) {
        ov::device::UUID uuid = {};
        std::copy_n(std::begin(device_info.uuid.val), cldnn::device_uuid::max_uuid_size, std::begin(uuid.uuid));
        return decltype(ov::device::uuid)::value_type {uuid};
    } else if (name == ov::device::full_name) {
        auto deviceName = StringRightTrim(device_info.dev_name, "NEO", false);
        deviceName += std::string(" (") + (device_info.dev_type == cldnn::device_type::discrete_gpu ? "dGPU" : "iGPU") + ")";
        return decltype(ov::device::full_name)::value_type {deviceName};
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, LegacyAPIHelper::get_supported_configs());
    } else if (name == ov::device::capabilities) {
        return decltype(ov::device::capabilities)::value_type {get_device_capabilities(device_info)};
    } else if (name == ov::range_for_async_infer_requests) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 2, 1);
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, range);
    } else if (name == ov::range_for_streams) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, device_info.num_ccs == 1 ? 2 : device_info.num_ccs);
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, range);
    } else if (name == GPU_METRIC_KEY(MEMORY_STATISTICS) ||
               name == ov::intel_gpu::memory_statistics) {
        std::map<std::string, uint64_t> statistics;
        for (auto const &item : statistics_map) {
            // Before collecting memory statistics of each context, it's updated with the latest memory statistics from engine.
            update_memory_statistics(item.first);
            for (auto const &kv : item.second) {
                if (!statistics.count(kv.first)) {
                    statistics[kv.first] = kv.second;
                } else {
                    statistics[kv.first] += kv.second;
                }
            }
        }
        return decltype(ov::intel_gpu::memory_statistics)::value_type {statistics};
    } else if (name == METRIC_KEY(MAX_BATCH_SIZE) ||
               name == ov::max_batch_size) {
        return decltype(ov::max_batch_size)::value_type {static_cast<uint32_t>(get_max_batch_size(options))};
    } else if (isModelCachingEnabled && name == METRIC_KEY(IMPORT_EXPORT_SUPPORT)) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    } else if (name == ov::caching_properties) {
        std::vector<ov::PropertyName> cachingProperties;
        cachingProperties.push_back(ov::PropertyName(ov::device::architecture.name(), PropertyMutability::RO));
        cachingProperties.push_back(ov::PropertyName(ov::intel_gpu::execution_units_count.name(), PropertyMutability::RO));
        cachingProperties.push_back(ov::PropertyName(ov::intel_gpu::driver_version.name(), PropertyMutability::RO));
        cachingProperties.push_back(ov::PropertyName(ov::inference_precision.name(), PropertyMutability::RW));
        cachingProperties.push_back(ov::PropertyName(ov::hint::execution_mode.name(), PropertyMutability::RW));
        return decltype(ov::caching_properties)::value_type(cachingProperties);
    } else if (name == ov::intel_gpu::driver_version) {
        return decltype(ov::intel_gpu::driver_version)::value_type {device_info.driver_version};
    } else if (name == ov::intel_gpu::device_id) {
        std::stringstream s;
        s << "0x" << std::hex << device_info.device_id;
        return decltype(ov::intel_gpu::device_id)::value_type {s.str()};
    } else if (name == ov::device::architecture) {
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
    } else {
        IE_THROW() << "Unsupported metric key " << name;
    }
}

std::vector<ov::PropertyName> Plugin::get_supported_properties() const {
    static const std::vector<ov::PropertyName> supported_properties = {
        // Metrics
        ov::PropertyName{ov::supported_properties.name(), PropertyMutability::RO},
        ov::PropertyName{ov::available_devices.name(), PropertyMutability::RO},
        ov::PropertyName{ov::range_for_async_infer_requests.name(), PropertyMutability::RO},
        ov::PropertyName{ov::range_for_streams.name(), PropertyMutability::RO},
        ov::PropertyName{ov::optimal_batch_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::max_batch_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::caching_properties.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::architecture.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::full_name.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::uuid.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::type.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::gops.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::capabilities.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::device_total_mem_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::uarch_version.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::execution_units_count.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::memory_statistics.name(), PropertyMutability::RO},

        // Configs
        ov::PropertyName{ov::enable_profiling.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::model_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::host_task_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::queue_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::queue_throttle.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::enable_loop_unrolling.name(), PropertyMutability::RW},
        ov::PropertyName{ov::cache_dir.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::compilation_num_threads.name(), PropertyMutability::RW},
        ov::PropertyName{ov::num_streams.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::num_requests.name(), PropertyMutability::RW},
        ov::PropertyName{ov::inference_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::device::id.name(), PropertyMutability::RW},
    };

    return supported_properties;
}

std::vector<std::string> Plugin::get_device_capabilities(const cldnn::device_info& info) const {
    std::vector<std::string> capabilities;

    capabilities.push_back(ov::device::capability::FP32);
    capabilities.push_back(ov::device::capability::BIN);
    if (!IsNewAPI())
        capabilities.push_back(METRIC_VALUE(BATCHED_BLOB));
    if (info.supports_fp16)
        capabilities.push_back(ov::device::capability::FP16);
    if (info.supports_imad || info.supports_immad)
        capabilities.push_back(ov::device::capability::INT8);
    if (info.supports_immad)
        capabilities.push_back(ov::intel_gpu::capability::HW_MATMUL);
    if (isModelCachingEnabled)
        capabilities.push_back(ov::device::capability::EXPORT_IMPORT);

    return capabilities;
}

uint32_t Plugin::get_max_batch_size(const std::map<std::string, Parameter>& options) const {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    auto device_id = GetConfig(ov::device::id.name(), options).as<std::string>();
    auto context = m_default_contexts.at(device_id)->get_impl();
    const auto& device_info = context->get_engine().get_device_info();
    const auto& config = m_configs_map.at(device_id);
    uint32_t n_streams = static_cast<uint32_t>(config.get_property(ov::num_streams));
    uint64_t occupied_device_mem = 0;
    auto statistic_result = GetMetric(ov::intel_gpu::memory_statistics.name(), options).as<std::map<std::string, uint64_t>>();
    auto occupied_usm_dev = statistic_result.find("usm_device_current");
    if (occupied_usm_dev != statistic_result.end()) {
        occupied_device_mem = occupied_usm_dev->second;
    }

    int64_t available_device_mem = device_info.max_global_mem_size - occupied_device_mem;
    GPU_DEBUG_LOG << "[GPU_MAX_BATCH_SIZE] available memory is " << available_device_mem
                  << " (occupied: " << occupied_device_mem << ")" << std::endl;

    int64_t max_batch_size = 1;

    if (options.find(ov::hint::model.name()) == options.end()) {
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] MODELS_PTR is not set: return 1" << std::endl;
        return static_cast<uint32_t>(max_batch_size);
    }

    auto it_streams = options.find("GPU_THROUGHPUT_STREAMS") != options.end() ? options.find("GPU_THROUGHPUT_STREAMS") :
                        options.find(ov::num_streams.name()) != options.end() ? options.find(ov::num_streams.name()) :
                        options.end();
    if (it_streams != options.end()) {
        if (it_streams->second.is<int32_t>()) {
            n_streams = it_streams->second.as<int32_t>();
        } else if (it_streams->second.is<uint32_t>()) {
            n_streams = it_streams->second.as<uint32_t>();
        } else if (it_streams->second.is<std::string>()) {
            auto n_streams_str = it_streams->second.as<std::string>();
            if (n_streams_str != CONFIG_VALUE(GPU_THROUGHPUT_AUTO) &&
                n_streams_str != util::to_string(ov::streams::AUTO)) {
                IE_THROW() << "[GPU_MAX_BATCH_SIZE] bad casting: GPU_THROUGHPUT_STREAMS should be either of uint32_t type or \"GPU_THROUGHPUT_AUTO\"";
            }
            n_streams = std::max(/* config.GetDefaultNStreamsForThroughputMode() */2u, device_info.num_ccs);
        } else {
            IE_THROW() << "[GPU_MAX_BATCH_SIZE] bad casting: GPU_THROUGHPUT_STREAMS should be either of uint32_t type or \"GPU_THROUGHPUT_AUTO\"";
        }
    }

    GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] n_streams : " << n_streams << std::endl;

    auto available_device_mem_it = options.find(ov::intel_gpu::hint::available_device_mem.name());
    if (available_device_mem_it != options.end()) {
        if (available_device_mem_it->second.is<int64_t>()) {
            available_device_mem = std::min(static_cast<int64_t>(available_device_mem), available_device_mem_it->second.as<int64_t>());
            GPU_DEBUG_LOG << "[GPU_MAX_BATCH_SIZE] available memory is reset by user " << available_device_mem << std::endl;
        } else {
            IE_THROW() << "[GPU_MAX_BATCH_SIZE] bad casting: ov::intel_gpu::hint::available_device_mem should be int64_t type";
        }
        if (available_device_mem < 0) {
            IE_THROW() << "[GPU_MAX_BATCH_SIZE] ov::intel_gpu::hint::available_device_mem value should be greater than 0 for max batch size calculation";
        }
    }

    std::shared_ptr<ngraph::Function> model;
    auto model_param = options.find(ov::hint::model.name())->second;
    if (model_param.is<std::shared_ptr<ngraph::Function>>()) {
        model = model_param.as<std::shared_ptr<ngraph::Function>>();
    } else {
        IE_THROW() << "[GPU_MAX_BATCH_SIZE] ov::hint::model should be std::shared_ptr<ov::Model> type";
    }

    InferenceEngine::CNNNetwork network(model);
    size_t base_batch_size = 16; // empirically decided for DG1

    auto& engine = get_default_context(device_id)->get_impl()->get_engine();

    std::shared_ptr<Program> program;

    GPU_DEBUG_IF(debug_config->base_batch_for_memory_estimation > 0) {
        size_t user_specified_base_batch_size = debug_config->base_batch_for_memory_estimation;
        base_batch_size = (user_specified_base_batch_size != base_batch_size) ? user_specified_base_batch_size : base_batch_size;
    }

    auto cloned_network = InferenceEngine::details::cloneNetwork(network);
    auto inputs_info = cloned_network.getInputsInfo();
    ICNNNetwork::InputShapes new_shapes;

    try {
        std::set<std::pair<std::string, size_t>> batched_inputs;

        auto function = InferenceEngine::details::cloneNetwork(cloned_network).getFunction();
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::FindBatch>(true, false);
        m.run_passes(function);
        const auto& params = function->get_parameters();
        for (size_t input_id = 0; input_id < params.size(); input_id++) {
            const auto& input = params[input_id];
            const auto& shape = input->get_partial_shape();
            // currently no plugin support batched execution for dynamic networks
            if (shape.is_dynamic()) {
                GPU_DEBUG_LOG << "[MAX_BATCH_SIZE] does not support dynamic networks" << std::endl;
                return static_cast<uint32_t>(max_batch_size);
            }

            if (shape.size()) {
                for (size_t s = 0; s < shape.size(); s++) {
                    if (ov::DimensionTracker::get_label(shape[s])) {
                        // batched dim for the input
                        auto batched_input_id = ov::op::util::get_ie_output_name(params[input_id]->output(0));
                        GPU_DEBUG_LOG << "[MAX_BATCH_SIZE] detected batched input " << batched_input_id
                                      << "[" << s << "]" << std::endl;
                        batched_inputs.insert(std::make_pair(batched_input_id, s));
                    }
                }
            }
        }

        if (!batched_inputs.size()) {
            GPU_DEBUG_LOG << "[MAX_BATCH_SIZE] MAX_BATCH_SIZE supports only networks with inputs/outputs featuring batched dim." << std::endl;
            return static_cast<uint32_t>(max_batch_size);
        }

        try {
            ICNNNetwork::InputShapes shapes = cloned_network.getInputShapes();
            for (const auto& input : batched_inputs)
                shapes[input.first][input.second] = base_batch_size;
            cloned_network.reshape(shapes);
        } catch (...) {
            GPU_DEBUG_INFO << "[MAX_BATCH_SIZE] Error at reshape to " << base_batch_size << std::endl;
            return static_cast<uint32_t>(max_batch_size);
        }

        auto nGraphFunc = cloned_network.getFunction();
        TransformationsPipeline transformations(config, device_info);
        transformations.apply(nGraphFunc);
        program = std::make_shared<Program>(cloned_network, engine, config, false, true);
        std::pair<int64_t, int64_t> device_memory_usage = program->GetCompiledProgram(0)->get_estimated_device_mem_usage();
        if (device_memory_usage.first == static_cast<int64_t>(-1L) && device_memory_usage.second == static_cast<int64_t>(-1L)) {
            return static_cast<uint32_t>(max_batch_size);
        }
        int64_t mem_for_general = std::max<int64_t>(1, available_device_mem - device_memory_usage.first);
        int64_t mem_per_batch = std::max<int64_t>(1, device_memory_usage.second / static_cast<int64_t>(base_batch_size));
        max_batch_size = mem_for_general / (mem_per_batch * static_cast<int64_t>(n_streams));
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Base batch size: " << base_batch_size  << std::endl;
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Const mem usage: " << device_memory_usage.first  << std::endl;
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] General mem usage: " << device_memory_usage.second  << std::endl;
    } catch (std::exception& e) {
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Failed in reshape or build program " << e.what() << std::endl;
    }

    return static_cast<uint32_t>(max_batch_size);
}

uint32_t Plugin::get_optimal_batch_size(const std::map<std::string, Parameter>& options) const {
    auto device_id = GetConfig(ov::device::id.name(), options).as<std::string>();
    auto context = m_default_contexts.at(device_id)->get_impl();
    const auto& device_info = context->get_engine().get_device_info();
    auto next_pow_of_2 = [] (float x) {
        return pow(2, ceil(std::log(x)/std::log(2)));
    };
    auto closest_pow_of_2 = [] (float x) {
        return pow(2, floor(std::log(x)/std::log(2)));
    };
    auto model_param = options.find(ov::hint::model.name());
    if (model_param == options.end()) {
        GPU_DEBUG_INFO << "[OPTIMAL_BATCH_SIZE] ov::hint::model is not set: return 1" << std::endl;
        return static_cast<uint32_t>(1);
    }
    std::shared_ptr<ngraph::Function> model;
    try {
        model = model_param->second.as<std::shared_ptr<ngraph::Function>>();
    } catch (...) {
        IE_THROW() << "[OPTIMAL_BATCH_SIZE] ov::hint::model should be std::shared_ptr<ov::Model> type";
    }
    GPU_DEBUG_INFO << "DEVICE_INFO:"
                   << "gfx_version.major, " << device_info.gfx_ver.major
                   << "gfx_version.minor " << std::to_string(device_info.gfx_ver.minor) << std::endl;
    static std::map<cldnn::gfx_version, size_t> gen_kbytes_per_bank = {
            {{12, 0, 0}, 480},  // TGL
            {{12, 1, 0}, 2048}, // DG1
            {{12, 5, 0}, 320},
            {{12, 7, 0}, 512},
    };
    size_t L3_cache_size = device_info.gfx_ver.major && (device_info.gfx_ver.major <= 9)
            ? 768 * 1024 // Gen9
            : 2 * 768 * 1024;  //reasonable default when no arch has been detected (e.g. due to old driver ver)
    cldnn::gfx_version gen = {device_info.gfx_ver.major, device_info.gfx_ver.minor, 0 /*ignore the revision*/};
    auto val = gen_kbytes_per_bank.find(gen);
    if (gen_kbytes_per_bank.end() != val) {
        auto kbytes_per_bank = val->second;
        auto num_banks_per_slice = device_info.num_sub_slices_per_slice > 4
                                    ? next_pow_of_2(device_info.num_sub_slices_per_slice)
                                    : 2 * device_info.num_sub_slices_per_slice;
        L3_cache_size = kbytes_per_bank * 1024 * num_banks_per_slice * device_info.num_slices;
        GPU_DEBUG_INFO << "DEVICE_INFO:"
                        << "num_slices " << device_info.num_slices
                        << ", num_sub_slices_per_slice " << device_info.num_sub_slices_per_slice
                        << ", num_banks_per_slice " << num_banks_per_slice
                        << ", gen_kbytes_per_bank : " << kbytes_per_bank
                        << ", L3_cache_size is (MB): " << float(L3_cache_size) / 1024 / 1024 << std::endl;
    }
    auto config = m_configs_map.at(device_id);
    auto networkCloned = clone_and_transform_model(CNNNetwork(model), config);
    ov::MemBandwidthPressure memPressure = ov::MemBandwidthPressureTolerance(networkCloned.getFunction(), L3_cache_size);
    uint32_t batch = 1;
    if (memPressure.max_mem_tolerance != ov::MemBandwidthPressure::UNKNOWN)
        batch = std::max(1.0, 16 * closest_pow_of_2(memPressure.max_mem_tolerance));
    std::map<std::string, InferenceEngine::Parameter> options_for_max_batch;
    options_for_max_batch[ov::hint::model.name()] = model;
    options_for_max_batch["GPU_THROUGHPUT_STREAMS"] = CONFIG_VALUE(GPU_THROUGHPUT_AUTO);
    auto max_batch_size = GetMetric(ov::max_batch_size.name(), options_for_max_batch).as<uint32_t>();
    uint32_t closest = closest_pow_of_2(max_batch_size);
    batch = std::min(closest, batch);
    batch = std::min(256u, batch); //batch 256 is a max
    GPU_DEBUG_INFO << memPressure.max_mem_tolerance << std::endl;
    GPU_DEBUG_INFO << "MAX_BATCH: " << max_batch_size << std::endl;
    GPU_DEBUG_INFO << "ACTUAL OPTIMAL BATCH: " << batch << std::endl;

    return batch;
}

}  // namespace intel_gpu
}  // namespace ov

static const Version version = { {2, 1}, CI_BUILD_NUMBER, "Intel GPU plugin" };
IE_DEFINE_PLUGIN_CREATE_FUNCTION(ov::intel_gpu::Plugin, version)
