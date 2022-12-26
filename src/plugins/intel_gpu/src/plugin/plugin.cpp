// Copyright (C) 2018-2022 Intel Corporation
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
#include "ie_plugin_config.hpp"
#include <ie_ngraph_utils.hpp>
#include <ie_algorithm.hpp>

#include "openvino/runtime/intel_gpu/properties.hpp"
#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/plugin/custom_layer.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "gpu/gpu_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_icore.hpp"

#include "dimension_tracker.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include <transformations/rt_info/fused_names_attribute.hpp>

#include <transformations/utils/utils.hpp>
#include "openvino/pass/serialize.hpp"
#include "openvino/pass/manager.hpp"
#include <ngraph/pass/manager.hpp>
#include <openvino/util/common_util.hpp>

#include "intel_gpu/runtime/device_query.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include <performance_heuristics.hpp>
#ifdef __linux__
# include <dlfcn.h>
#endif

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

void Plugin::RegisterPrimitives() {
    #define REGISTER_FACTORY(op_version, op_name) FACTORY_CALL(op_version, op_name)
    #include "intel_gpu/plugin/primitives_list.hpp"
    #undef REGISTER_FACTORY
}

struct Plugin::impl {
    Configs m_configs;
};

std::string Plugin::get_device_id_from_config(const std::map<std::string, std::string>& config) const {
    std::string device_id;
    if (config.find(PluginConfigParams::KEY_DEVICE_ID) != config.end()) {
        device_id = config.at(PluginConfigParams::KEY_DEVICE_ID);
    }
    return device_id;
}

cldnn::device_info Plugin::GetDeviceInfo(const std::map<std::string, std::string> &config) const {
    auto device_info = device_map.begin()->second->get_info();
    std::string device_id = get_device_id_from_config(config);
    if (!device_id.empty()) {
        if (device_map.find(device_id) == device_map.end()) {
            IE_THROW() << "Invalid device ID: " << device_id;
        }
        device_info = device_map.at(device_id)->get_info();
    }

    return device_info;
}

void Plugin::TransformNetwork(std::shared_ptr<ov::Model>& model, const Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::TransformNetwork");
    auto deviceInfo = GetDeviceInfo(config.key_config_map);
    TransformationsPipeline transformations(config, deviceInfo);
    transformations.apply(model);
}

InferenceEngine::CNNNetwork Plugin::CloneAndTransformNetwork(const InferenceEngine::CNNNetwork& network,
                                                             const Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::CloneAndTransformNetwork");
    GPU_DEBUG_DEFINE_MEM_LOGGER("Plugin::CloneAndTransformNetwork");
    CNNNetwork clonedNetwork = InferenceEngine::details::cloneNetwork(network);

    auto nGraphFunc = clonedNetwork.getFunction();
    if (nGraphFunc) {
        TransformNetwork(nGraphFunc, config);
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(!debug_config->dump_graphs.empty()) {
            auto path_base = debug_config->dump_graphs + "/" + network.getName() + "_" +  "transformed_func";
            ov::pass::Serialize(path_base + ".xml", path_base + ".bin").run_on_model(nGraphFunc);
    }
    }
    return clonedNetwork;
}

Plugin::Plugin() : m_defaultContexts({}) {
    _pluginName = "GPU";
    _impl = std::make_shared<impl>();
    RegisterPrimitives();
    // try loading gpu engine and get info from it
    {
        // Set OCL runtime which should be always available
        cldnn::device_query device_query(cldnn::engine_types::ocl, cldnn::runtime_types::ocl);
        device_map = device_query.get_available_devices();

        // Set default configs for each device
        for (auto& device : device_map) {
            _impl->m_configs.CreateConfig(device.first);
        }
    }
    // locate global custom kernel config
    // and auto-load kernels from it
#ifdef _WIN32
    CHAR mpath[MAX_PATH + 1];
    HMODULE nModule;
    GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)CustomLayer::LoadFromFile,
        &nModule);
    GetModuleFileName(nModule, mpath, sizeof(mpath));
#elif __linux__
    Dl_info dl_info;
    dladdr(reinterpret_cast<void *>(CustomLayer::LoadFromFile), &dl_info);
    const char* mpath = dl_info.dli_fname;
#endif
    std::string configFile(mpath);
    std::size_t dir_split_pos = configFile.find_last_of("/\\");
    std::string config_path;

    if (dir_split_pos != std::string::npos) {
        // path contains directory
        config_path = configFile.substr(0, dir_split_pos);
    }
    config_path += "/cldnn_global_custom_kernels/cldnn_global_custom_kernels.xml";
    for (auto& config : _impl->m_configs) {
        CustomLayer::LoadFromFile(config_path, config.second.customLayers, true);
    }

    if (const char* env_p = std::getenv("OV_GPU_CACHE_MODEL")) {
        if (env_p[0] == '1') {
            isModelCachingEnabled = true;
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

void Plugin::UpdateConfig(Config& conf, const InferenceEngine::CNNNetwork &network, const std::map<std::string, std::string> &params) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::UpdateConfig");
    auto device_info = GetDeviceInfo(params);
    conf.enableInt8 = device_info.supports_imad || device_info.supports_immad;
    conf.UpdateFromMap(params, device_info);
    if (conf.enableDynamicBatch) {
        conf.max_dynamic_batch = static_cast<int>(network.getBatchSize());
    }
}

void Plugin::UpdateStatistics(const RemoteContextImpl::Ptr& context) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::UpdateStatistics");
    {
        std::lock_guard<std::mutex> lock(engine_mutex);

        // if the same context exists, the statistics is replaced with the latest one
        // (currently, memory usage is accumulated for several networks in the same context)
        // if it does not exist, a new statistics is added
        statistics_map[context] = context->get_engine().get_memory_statistics();
    }
}

std::map<std::string, std::string> Plugin::ConvertPerfHintsToConfig(
        const std::map<std::string, std::string>& network_config,
        const Config& plugin_config) const {
    // deduces the actual settings from the performance hints and returns fully-defined config
    auto config = network_config;
    const auto &mode = config.find(PluginConfigParams::KEY_PERFORMANCE_HINT);
    // the mode may have just arrived to the LoadNetwork, or was set with the plugins' SetConfig
    if (mode != config.end() || !plugin_config.perfHintsConfig.ovPerfHint.empty()) {
        const auto mode_name = (mode != config.end())
                               ? PerfHintsConfig::CheckPerformanceHintValue(mode->second)
                               : plugin_config.perfHintsConfig.ovPerfHint;
        //checking streams (to avoid overriding what user might explicitly set in the incoming config or previously via SetConfig)
        const auto streams = config.find(PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS) == config.end() &&
                             config.find(ov::num_streams.name()) == config.end();
        if (streams && !streamsSet) {
            if (mode_name == CONFIG_VALUE(LATENCY)) {
                config[PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS] = std::to_string(1);
                config[ov::num_streams.name()] = std::to_string(1);
            } else if (mode_name == CONFIG_VALUE(THROUGHPUT)) {
                config[PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS] = CONFIG_VALUE(GPU_THROUGHPUT_AUTO);
                config[ov::num_streams.name()] = ov::util::to_string(ov::streams::AUTO);
                //disabling the throttling temporarily to set the validation (that is switching to the hints) perf baseline
                //checking throttling (to avoid overriding what user might explicitly set in the incoming config or previously via SetConfig)
                // const auto bInConfig = config.find(GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE) != config.end() ||
                //    config.find(CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE) != config.end();
                // if (!bInConfig && !throttlingSet)
                //    config[GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE] = std::to_string(1);
            }
        }
    }
    return config;
}

IExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                           const std::map<std::string, std::string> &orig_config) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::LoadExeNetworkImpl");
    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs = network.getInputsInfo();
    check_inputs(_networkInputs);

    Configs confs = _impl->m_configs;
    std::string device_id = get_device_id_from_config(orig_config);
    Config conf = confs.GetConfig(device_id);

    auto config = ConvertPerfHintsToConfig(orig_config, conf);
    UpdateConfig(conf, network, config);

    ExecutionConfig new_conf(ov::AnyMap(conf.key_config_map.begin(), conf.key_config_map.end()));

    auto context = get_default_context(conf);

    auto transformedNetwork = CloneAndTransformNetwork(network, conf);
    {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::LoadExeNetworkImpl::CreateExeNetwork");
        CompiledModel::Ptr exeNetwork = std::make_shared<CompiledModel>(transformedNetwork, context, conf, new_conf);
        UpdateStatistics(context->get_impl());
        return exeNetwork;
    }
}

IExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                           const InferenceEngine::RemoteContext::Ptr &context,
                                                           const std::map<std::string, std::string> &orig_config) {
    InferenceEngine::InputsDataMap _networkInputs = network.getInputsInfo();
    check_inputs(_networkInputs);

    auto casted = std::dynamic_pointer_cast<ClContext>(context);
    if (nullptr == casted) {
        IE_THROW() << "Invalid context";
    }

    Config conf = get_context_impl(casted)->get_config();
    auto config = ConvertPerfHintsToConfig(orig_config, conf);
    UpdateConfig(conf, network, config);

    ExecutionConfig new_conf(ov::AnyMap(conf.key_config_map.begin(), conf.key_config_map.end()));

    auto transformedNetwork = CloneAndTransformNetwork(network, conf);
    return std::make_shared<CompiledModel>(transformedNetwork, casted, conf, new_conf);
}

InferenceEngine::RemoteContext::Ptr Plugin::CreateContext(const AnyMap& params) {
    std::string context_type = extract_object<std::string>(params, GPU_PARAM_KEY(CONTEXT_TYPE));

    if (GPU_PARAM_VALUE(OCL) == context_type) {
        return std::make_shared<RemoteCLContext>(GetName(), params, _impl->m_configs.GetDefaultDeviceConfig());
    } else if (GPU_PARAM_VALUE(VA_SHARED) == context_type) {
#ifdef _WIN32
        return std::make_shared<RemoteD3DContext>(GetName(), params, _impl->m_configs.GetDefaultDeviceConfig());
#else
        return std::make_shared<RemoteVAContext>(GetName(), params, _impl->m_configs.GetDefaultDeviceConfig());
#endif
    }

    OPENVINO_ASSERT(false, "[GPU] Unsupported context type passed to CreateContext method: ", context_type);
}

RemoteCLContext::Ptr Plugin::get_default_context(const Config& config) const {
    if (m_defaultContexts.find(config.device_id) == m_defaultContexts.end()) {
        m_defaultContexts[config.device_id] = std::make_shared<RemoteCLContext>(GetName(), AnyMap(), config);
    }

    return m_defaultContexts.at(config.device_id);;
}

InferenceEngine::RemoteContext::Ptr Plugin::GetDefaultContext(const AnyMap& params) {
    std::string device_id = "";

    if (params.find(CONFIG_KEY(DEVICE_ID)) != params.end())
        device_id = params.at(CONFIG_KEY(DEVICE_ID)).as<std::string>();

    const Config conf = _impl->m_configs.GetConfig(device_id);

    return get_default_context(conf);
}

void Plugin::SetConfig(const std::map<std::string, std::string> &config) {
    streamsSet = config.find(PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS) != config.end() ||
                 config.find(ov::num_streams.name()) != config.end();
    throttlingSet = config.find(GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE) != config.end() ||
                    config.find(ov::intel_gpu::hint::queue_throttle.name()) != config.end();
    std::string device_id;
    cldnn::device_info device_info = device_map.begin()->second->get_info();
    if (config.find(PluginConfigInternalParams::KEY_CONFIG_DEVICE_ID) != config.end()) {
        device_id = config.at(PluginConfigInternalParams::KEY_CONFIG_DEVICE_ID);
        if (!device_id.empty() && device_map.find(device_id) != device_map.end()) {
            device_info = device_map.at(device_id)->get_info();
        }
        _impl->m_configs.GetConfig(device_id).UpdateFromMap(config, device_info);
    } else {
        device_id = get_device_id_from_config(config);
        if (!device_id.empty()) {
            if (device_map.find(device_id) != device_map.end()) {
                device_info = device_map.at(device_id)->get_info();
            }
            _impl->m_configs.SetDefaultDeviceID(device_id);
            _impl->m_configs.GetConfig(device_id).UpdateFromMap(config, device_info);
        } else {
            for (auto& conf : _impl->m_configs) {
                if (device_map.find(conf.first) != device_map.end()) {
                    device_info = device_map.at(conf.first)->get_info();
                }
                conf.second.UpdateFromMap(config, device_info);
            }
        }
    }
}

QueryNetworkResult Plugin::QueryNetwork(const CNNNetwork& network,
                                        const std::map<std::string, std::string>& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::QueryNetwork");
    QueryNetworkResult res;
    Configs confs = _impl->m_configs;
    std::string device_id = get_device_id_from_config(config);
    Config conf = confs.GetConfig(device_id);

    UpdateConfig(conf, network, config);

    auto ctx = get_default_context(conf)->get_impl();
    Program prog(ctx->get_engine(), conf);
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
        TransformNetwork(model, conf);
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
        res.supportedLayersMap.emplace(layerName, GetName() + "." + conf.device_id);
    }

    return res;
}

InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::ImportNetwork(std::istream& networkModel,
                                                                       const std::map<std::string, std::string>& orig_config) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::ImportNetwork");
    Configs confs = _impl->m_configs;
    std::string device_id = get_device_id_from_config(orig_config);
    Config conf = confs.GetConfig(device_id);

    auto config = ConvertPerfHintsToConfig(orig_config, conf);

    auto context = get_default_context(conf);

    ExecutionConfig new_conf(ov::AnyMap(conf.key_config_map.begin(), conf.key_config_map.end()));
    {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::ImportNetwork::CreateExeNetwork");
        CompiledModel::Ptr exeNetwork = std::make_shared<CompiledModel>(networkModel, context, conf, new_conf);
        exeNetwork->SetPointerToPlugin(shared_from_this());
        UpdateStatistics(context->get_impl());
        return exeNetwork;
    }
}

Parameter Plugin::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::GetConfig");
    Parameter result;

    std::string device_id;
    if (options.find(ov::device::id.name()) != options.end()) {
        device_id = options.find(ov::device::id.name())->second.as<std::string>();
    }
    Config config = _impl->m_configs.GetConfig(device_id);

    const bool is_new_api = IsNewAPI();
    if (config.key_config_map.find(name) != config.key_config_map.end()) {
        std::string val = config.key_config_map.find(name)->second;
        if (is_new_api) {
            if (name == ov::enable_profiling) {
                return val == PluginConfigParams::YES ? true : false;
            } else if (name == ov::hint::model_priority) {
                return ov::util::from_string(val, ov::hint::model_priority);
            } else if (name == ov::intel_gpu::hint::host_task_priority) {
                return ov::util::from_string(val, ov::intel_gpu::hint::host_task_priority);
            } else if (name == ov::intel_gpu::hint::queue_priority) {
                return ov::util::from_string(val, ov::intel_gpu::hint::queue_priority);
            } else if (name == ov::intel_gpu::hint::queue_throttle) {
                return ov::util::from_string(val, ov::intel_gpu::hint::queue_throttle);
            } else if (name == ov::intel_gpu::enable_loop_unrolling) {
                return val == PluginConfigParams::YES ? true : false;
            } else if (name == ov::cache_dir) {
                return ov::util::from_string(val, ov::cache_dir);
            } else if (name == ov::hint::performance_mode) {
                return ov::util::from_string(val, ov::hint::performance_mode);
            } else if (name == ov::compilation_num_threads) {
                return ov::util::from_string(val, ov::compilation_num_threads);
            } else if (name == ov::num_streams) {
                return ov::util::from_string(val, ov::num_streams);
            } else if (name == ov::hint::num_requests) {
                return ov::util::from_string(val, ov::hint::num_requests);
            } else if (name == ov::hint::inference_precision) {
                return ov::util::from_string(val, ov::hint::inference_precision);
            } else if (name == ov::device::id) {
                return ov::util::from_string(val, ov::device::id);
            } else {
                return val;
            }
        } else {
            if (name == PluginConfigParams::KEY_MODEL_PRIORITY ||
                name == GPUConfigParams::KEY_GPU_HOST_TASK_PRIORITY)
                return Config::ConvertPropertyToLegacy(name, val);
            else
                return val;
        }
    } else {
        IE_THROW() << "3-Unsupported config key : " << name;
    }
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
    std::string device_id = GetConfig(ov::device::id.name(), options);

    auto iter = device_map.find(std::to_string(cldnn::device_query::device_id));
    if (iter == device_map.end())
        iter = device_map.find(device_id);
    if (iter == device_map.end())
        iter = device_map.begin();
    auto device = iter->second;
    auto device_info = device->get_info();
    bool is_new_api = IsNewAPI();

    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {
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
            ov::PropertyName{ov::compilation_num_threads.name(), PropertyMutability::RW},
            ov::PropertyName{ov::num_streams.name(), PropertyMutability::RW},
            ov::PropertyName{ov::hint::num_requests.name(), PropertyMutability::RW},
            ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RW},
            ov::PropertyName{ov::device::id.name(), PropertyMutability::RW},
        };
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(AVAILABLE_DEVICES));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_STREAMS));
        metrics.push_back(METRIC_KEY(DEVICE_TYPE));
        metrics.push_back(METRIC_KEY(DEVICE_GOPS));
        metrics.push_back(METRIC_KEY(OPTIMAL_BATCH_SIZE));
        metrics.push_back(METRIC_KEY(MAX_BATCH_SIZE));
        if (isModelCachingEnabled)
            metrics.push_back(METRIC_KEY(IMPORT_EXPORT_SUPPORT));
        metrics.push_back(GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
        metrics.push_back(GPU_METRIC_KEY(UARCH_VERSION));
        metrics.push_back(GPU_METRIC_KEY(EXECUTION_UNITS_COUNT));
        metrics.push_back(GPU_METRIC_KEY(MEMORY_STATISTICS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
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
        auto next_pow_of_2 = [] (float x) {
            return pow(2, ceil(std::log(x)/std::log(2)));
        };
        auto closest_pow_of_2 = [] (float x) {
            return pow(2, floor(std::log(x)/std::log(2)));
        };
        auto model_param = options.find(ov::hint::model.name());
        if (model_param == options.end()) {
            GPU_DEBUG_INFO << "[OPTIMAL_BATCH_SIZE] ov::hint::model is not set: return 1" << std::endl;
            return decltype(ov::optimal_batch_size)::value_type {static_cast<unsigned int>(1)};
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
        Config config = _impl->m_configs.GetConfig(device_id);
        auto networkCloned = CloneAndTransformNetwork(CNNNetwork(model), config);
        ov::MemBandwidthPressure memPressure = ov::MemBandwidthPressureTolerance(networkCloned.getFunction(), L3_cache_size);
        unsigned int batch = 1;
        if (memPressure.max_mem_tolerance != ov::MemBandwidthPressure::UNKNOWN)
            batch = std::max(1.0, 16 * closest_pow_of_2(memPressure.max_mem_tolerance));
        std::map<std::string, InferenceEngine::Parameter> options_for_max_batch;
        options_for_max_batch[ov::hint::model.name()] = model;
        options_for_max_batch["GPU_THROUGHPUT_STREAMS"] = CONFIG_VALUE(GPU_THROUGHPUT_AUTO);
        auto max_batch_size = GetMetric(ov::max_batch_size.name(), options_for_max_batch).as<unsigned int>();
        unsigned int closest = closest_pow_of_2(max_batch_size);
        batch = std::min(closest, batch);
        batch = std::min(256u, batch); //batch 256 is a max
        GPU_DEBUG_INFO << memPressure.max_mem_tolerance << std::endl;
        GPU_DEBUG_INFO << "MAX_BATCH: " << max_batch_size << std::endl;
        GPU_DEBUG_INFO << "ACTUAL OPTIMAL BATCH: " << batch << std::endl;
        return decltype(ov::optimal_batch_size)::value_type {batch};
    } else if (name == ov::device::uuid) {
        ov::device::UUID uuid = {};
        std::copy_n(std::begin(device_info.uuid.val), cldnn::device_uuid::max_uuid_size, std::begin(uuid.uuid));
        return decltype(ov::device::uuid)::value_type {uuid};
    } else if (name == ov::device::full_name) {
        auto deviceName = StringRightTrim(device_info.dev_name, "NEO", false);
        deviceName += std::string(" (") + (device_info.dev_type == cldnn::device_type::discrete_gpu ? "dGPU" : "iGPU") + ")";
        return decltype(ov::device::full_name)::value_type {deviceName};
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto opt : _impl->m_configs.GetConfig(device_id).key_config_map) {
            // Exclude new API properties
            if (!Config::isNewApiProperty(opt.first))
                configKeys.push_back(opt.first);
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == ov::device::capabilities) {
        std::vector<std::string> capabilities;

        capabilities.push_back(ov::device::capability::FP32);
        capabilities.push_back(ov::device::capability::BIN);
        if (!is_new_api)
            capabilities.push_back(METRIC_VALUE(BATCHED_BLOB));
        if (device_info.supports_fp16)
            capabilities.push_back(ov::device::capability::FP16);
        if (device_info.supports_imad || device_info.supports_immad)
            capabilities.push_back(ov::device::capability::INT8);
        if (device_info.supports_immad)
            capabilities.push_back(ov::intel_gpu::capability::HW_MATMUL);
        if (isModelCachingEnabled)
            capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
        return decltype(ov::device::capabilities)::value_type {capabilities};
    } else if (name == ov::range_for_async_infer_requests) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 2, 1);
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, range);
    } else if (name == ov::range_for_streams) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, 2);
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, range);
    } else if (name == GPU_METRIC_KEY(MEMORY_STATISTICS) ||
               name == ov::intel_gpu::memory_statistics) {
        std::map<std::string, uint64_t> statistics;
        for (auto const &item : statistics_map) {
            // Before collecting memory statistics of each context, it's updated with the latest memory statistics from engine.
            UpdateStatistics(item.first);
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
        const auto& config = _impl->m_configs.GetConfig(device_id);
        uint32_t n_streams = static_cast<uint32_t>(config.throughput_streams);
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
            return decltype(ov::max_batch_size)::value_type {static_cast<uint32_t>(max_batch_size)};
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
                std::string n_streams_str = it_streams->second.as<std::string>();
                if (n_streams_str != CONFIG_VALUE(GPU_THROUGHPUT_AUTO) &&
                    n_streams_str != util::to_string(ov::streams::AUTO)) {
                    IE_THROW() << "[GPU_MAX_BATCH_SIZE] bad casting: GPU_THROUGHPUT_STREAMS should be either of uint32_t type or \"GPU_THROUGHPUT_AUTO\"";
                }
                n_streams = std::max(config.GetDefaultNStreamsForThroughputMode(), device_info.num_ccs);
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

        ExecutionConfig new_conf(ov::AnyMap(config.key_config_map.begin(), config.key_config_map.end()));

        auto& engine = get_default_context(config)->get_impl()->get_engine();

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
            m.register_pass<ngraph::pass::InitNodeInfo>();
            m.register_pass<ov::pass::FindBatch>(true, false);
            m.run_passes(function);
            const auto& params = function->get_parameters();
            for (size_t input_id = 0; input_id < params.size(); input_id++) {
                const auto& input = params[input_id];
                const auto& shape = input->get_partial_shape();
                // currently no plugin support batched execution for dynamic networks
                if (shape.is_dynamic()) {
                    GPU_DEBUG_LOG << "[MAX_BATCH_SIZE] does not support dynamic networks" << std::endl;
                    return decltype(ov::max_batch_size)::value_type {static_cast<uint32_t>(max_batch_size)};
                }

                if (shape.size()) {
                    for (size_t s = 0; s < shape.size(); s++) {
                        if (ov::DimensionTracker::get_label(shape[s])) {
                            // batched dim for the input
                            auto batched_input_id = ngraph::op::util::get_ie_output_name(params[input_id]->output(0));
                            GPU_DEBUG_LOG << "[MAX_BATCH_SIZE] detected batched input " << batched_input_id
                                          << "[" << s << "]" << std::endl;
                            batched_inputs.insert(std::make_pair(batched_input_id, s));
                        }
                    }
                }
            }

            if (!batched_inputs.size()) {
                GPU_DEBUG_LOG << "[MAX_BATCH_SIZE] MAX_BATCH_SIZE supports only networks with inputs/outputs featuring batched dim." << std::endl;
                return decltype(ov::max_batch_size)::value_type {static_cast<uint32_t>(max_batch_size)};
            }

            try {
                ICNNNetwork::InputShapes shapes = cloned_network.getInputShapes();
                for (const auto& input : batched_inputs)
                    shapes[input.first][input.second] = base_batch_size;
                cloned_network.reshape(shapes);
            } catch (...) {
                GPU_DEBUG_INFO << "[MAX_BATCH_SIZE] Error at reshape to " << base_batch_size << std::endl;
                return decltype(ov::max_batch_size)::value_type {static_cast<uint32_t>(max_batch_size)};
            }

            auto nGraphFunc = cloned_network.getFunction();
            TransformationsPipeline transformations(config, device_info);
            transformations.apply(nGraphFunc);
            program = std::make_shared<Program>(cloned_network, engine, config, new_conf, false, true);
            std::pair<int64_t, int64_t> device_memory_usage = program->GetCompiledProgram(0)->get_estimated_device_mem_usage();
            if (device_memory_usage.first == static_cast<int64_t>(-1L) && device_memory_usage.second == static_cast<int64_t>(-1L)) {
                return decltype(ov::max_batch_size)::value_type {static_cast<uint32_t>(max_batch_size)};
            }
            int64_t mem_for_general = std::max(static_cast<int64_t>(1L),
                    static_cast<int64_t>(static_cast<int64_t>(available_device_mem) - device_memory_usage.first));
            int64_t mem_per_batch = std::max(static_cast<int64_t>(1L), (device_memory_usage.second / static_cast<int64_t>(base_batch_size)));
            max_batch_size = mem_for_general / (mem_per_batch * static_cast<int64_t>(n_streams));
            GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Base batch size: " << base_batch_size  << std::endl;
            GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Const mem usage: " << device_memory_usage.first  << std::endl;
            GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] General mem usage: " << device_memory_usage.second  << std::endl;
        } catch (std::exception& e) {
            GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Failed in reshape or build program " << e.what() << std::endl;
        }
        return decltype(ov::max_batch_size)::value_type {static_cast<uint32_t>(max_batch_size)};
    } else if (isModelCachingEnabled && name == METRIC_KEY(IMPORT_EXPORT_SUPPORT)) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    } else if (name == ov::caching_properties) {
        std::vector<ov::PropertyName> cachingProperties;
        cachingProperties.push_back(ov::PropertyName(ov::intel_gpu::uarch_version.name(), PropertyMutability::RO));
        cachingProperties.push_back(ov::PropertyName(ov::intel_gpu::execution_units_count.name(), PropertyMutability::RO));
        cachingProperties.push_back(ov::PropertyName(ov::intel_gpu::driver_version.name(), PropertyMutability::RO));
        cachingProperties.push_back(ov::PropertyName(ov::intel_gpu::device_id.name(), PropertyMutability::RO));
        return decltype(ov::caching_properties)::value_type(cachingProperties);
    } else if (name == ov::intel_gpu::driver_version) {
        return decltype(ov::intel_gpu::driver_version)::value_type {device_info.driver_version};
    } else if (name == ov::intel_gpu::device_id) {
        std::stringstream s;
        s << "0x" << std::hex << device_info.device_id;
        return decltype(ov::intel_gpu::device_id)::value_type {s.str()};
    } else if (name == ov::device::architecture) {
        std::stringstream s;
        s << "GPU: ";
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
}  // namespace intel_gpu
}  // namespace ov

static const Version version = { {2, 1}, CI_BUILD_NUMBER, "Intel GPU plugin" };
IE_DEFINE_PLUGIN_CREATE_FUNCTION(ov::intel_gpu::Plugin, version)
