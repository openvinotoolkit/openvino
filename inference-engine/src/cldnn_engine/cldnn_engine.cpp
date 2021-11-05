// Copyright (C) 2018-2021 Intel Corporation
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

#include "cldnn_engine.h"
#include "cldnn_executable_network.h"
#include "cldnn_transformations_pipeline.h"
#include "cldnn_custom_layer.h"
#include "cldnn_itt.h"
#include "gpu/gpu_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

#include <transformations/rt_info/fused_names_attribute.hpp>

#include "cldnn/runtime/device_query.hpp"
#include "cldnn/runtime/debug_configuration.hpp"

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

namespace CLDNNPlugin {

#define FACTORY_DECLARATION(op_version, op_name) \
    void __register ## _ ## op_name ## _ ## op_version();

#define FACTORY_CALL(op_version, op_name) \
    __register ## _ ## op_name ## _ ## op_version();

#define REGISTER_FACTORY(op_version, op_name) FACTORY_DECLARATION(op_version, op_name)
#include "cldnn_primitives_list.hpp"
#undef REGISTER_FACTORY

void clDNNEngine::RegisterPrimitives() {
    #define REGISTER_FACTORY(op_version, op_name) FACTORY_CALL(op_version, op_name)
    #include "cldnn_primitives_list.hpp"
    #undef REGISTER_FACTORY
}

struct clDNNEngine::impl {
    CLDNNPlugin::Configs m_configs;
};

std::string clDNNEngine::GetDeviceIDFromConfig(const std::map<std::string, std::string>& config) const {
    std::string device_id;
    if (config.find(PluginConfigParams::KEY_DEVICE_ID) != config.end()) {
        device_id = config.at(PluginConfigParams::KEY_DEVICE_ID);
    }
    return device_id;
}

cldnn::device_info clDNNEngine::GetDeviceInfo(const std::map<std::string, std::string> &config) const {
    auto device_info = device_map.begin()->second->get_info();
    std::string device_id = GetDeviceIDFromConfig(config);
    if (!device_id.empty()) {
        if (device_map.find(device_id) == device_map.end()) {
            IE_THROW() << "Invalid device ID: " << device_id;
        }
        device_info = device_map.at(device_id)->get_info();
    }

    return device_info;
}

InferenceEngine::CNNNetwork clDNNEngine::CloneAndTransformNetwork(const InferenceEngine::CNNNetwork& network,
                                                                  const CLDNNPlugin::Config& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::CloneAndTransformNetwork");
    CNNNetwork clonedNetwork = InferenceEngine::details::cloneNetwork(network);

    if (clonedNetwork.getFunction()) {
        auto nGraphFunc = clonedNetwork.getFunction();
        auto deviceInfo = GetDeviceInfo(config.key_config_map);
        TransformationsPipeline transformations(config, deviceInfo);
        transformations.apply(nGraphFunc);
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->dump_graphs.empty()) {
        clonedNetwork.serialize(debug_config->dump_graphs + "/" + network.getName() + "_" +  "transformed_func.xml");
    }
    return clonedNetwork;
}

clDNNEngine::clDNNEngine() : m_defaultContext(nullptr) {
    _pluginName = "GPU";
    _impl = std::make_shared<impl>();
    RegisterPrimitives();
    // try loading clDNN engine and get info from it
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
        (LPCSTR)CLDNNCustomLayer::LoadFromFile,
        &nModule);
    GetModuleFileName(nModule, mpath, sizeof(mpath));
#elif __linux__
    Dl_info dl_info;
    dladdr(reinterpret_cast<void *>(CLDNNCustomLayer::LoadFromFile), &dl_info);
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
        CLDNNCustomLayer::LoadFromFile(config_path, config.second.customLayers, true);
    }
}

auto check_inputs = [](InferenceEngine::InputsDataMap _networkInputs) {
    for (auto ii : _networkInputs) {
        auto input_precision = ii.second->getTensorDesc().getPrecision();
        if (input_precision != InferenceEngine::Precision::FP16 &&
            input_precision != InferenceEngine::Precision::FP32 &&
            input_precision != InferenceEngine::Precision::U8 &&
            input_precision != InferenceEngine::Precision::I8 &&
            input_precision != InferenceEngine::Precision::I16 &&
            input_precision != InferenceEngine::Precision::U16 &&
            input_precision != InferenceEngine::Precision::I32 &&
            input_precision != InferenceEngine::Precision::I64 &&
            input_precision != InferenceEngine::Precision::BOOL) {
            IE_THROW(NotImplemented)
                << "Input image format " << input_precision << " is not supported yet...";
        }
    }
};

void clDNNEngine::UpdateConfig(CLDNNPlugin::Config& conf, const InferenceEngine::CNNNetwork &network, const std::map<std::string, std::string> &params) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::UpdateConfig");
    auto device_info = GetDeviceInfo(params);
    conf.enableInt8 = device_info.supports_imad || device_info.supports_immad;
    conf.UpdateFromMap(params);
    if (conf.enableDynamicBatch) {
        conf.max_dynamic_batch = static_cast<int>(network.getBatchSize());
    }
}

void clDNNEngine::UpdateStatistics(CLDNNRemoteCLContext::Ptr context) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::UpdateStatistics");
    {
        std::lock_guard<std::mutex> lock(engine_mutex);

        std::map<std::string, uint64_t> statistics;
        auto impl = getContextImpl(context);
        impl->acquire_lock();
        std::shared_ptr<cldnn::engine> eng = impl->GetEngine();
        statistics = eng->get_memory_statistics();
        impl->release_lock();

        // if the same context exists, the statistics is replaced with the latest one
        // (currently, memory usage is accumulated for several networks in the same context)
        // if it does not exist, a new statistics is added
        statistics_map[context] = statistics;
    }
}

std::map<std::string, std::string> clDNNEngine::ConvertPerfHintsToConfig(
        const std::map<std::string, std::string>& network_config,
        const CLDNNPlugin::Config& plugin_config) const {
    // deduces the actual settings from the performance hints and returns fully-defined config
    auto config = network_config;
    const auto &mode = config.find(PluginConfigParams::KEY_PERFORMANCE_HINT);
    // the mode may have just arrived to the LoadNetwork, or was set with the plugins' SetConfig
    if (mode != config.end() || !plugin_config.perfHintsConfig.ovPerfHint.empty()) {
        const auto mode_name = (mode != config.end())
                               ? PerfHintsConfig::CheckPerformanceHintValue(mode->second)
                               : plugin_config.perfHintsConfig.ovPerfHint;
        //checking streams (to avoid overriding what user might explicitly set in the incoming config or previously via SetConfig)
        const auto streams = config.find(PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS);
        if (streams == config.end() && !streamsSet) {
            if (mode_name == CONFIG_VALUE(LATENCY)) {
                config[PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS] = std::to_string(1);
            } else if (mode_name == CONFIG_VALUE(THROUGHPUT)) {
                config[PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS] = CONFIG_VALUE(GPU_THROUGHPUT_AUTO);
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

IExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                                const std::map<std::string, std::string> &orig_config) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::LoadExeNetworkImpl");
    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs = network.getInputsInfo();
    check_inputs(_networkInputs);

    CLDNNPlugin::Configs confs = _impl->m_configs;
    std::string device_id = GetDeviceIDFromConfig(orig_config);
    CLDNNPlugin::Config conf = confs.GetConfig(device_id);

    auto config = ConvertPerfHintsToConfig(orig_config, conf);
    UpdateConfig(conf, network, config);

    CLDNNRemoteCLContext::Ptr context;

    auto canReuseDefaultContext = [&]() -> bool {
        if (m_defaultContext == nullptr)
            return false;

        const Config& context_config = m_defaultContext->GetConfig();
        const Config& current_config = conf;

        return context_config.throughput_streams == current_config.throughput_streams &&
               context_config.useProfiling == current_config.useProfiling &&
               context_config.dumpCustomKernels == current_config.dumpCustomKernels &&
               context_config.memory_pool_on == current_config.memory_pool_on &&
               context_config.queueThrottle == current_config.queueThrottle &&
               context_config.queuePriority == current_config.queuePriority &&
               context_config.sources_dumps_dir == current_config.sources_dumps_dir &&
               context_config.tuningConfig.mode == current_config.tuningConfig.mode &&
               context_config.tuningConfig.cache_file_path == current_config.tuningConfig.cache_file_path &&
               context_config.kernels_cache_dir == current_config.kernels_cache_dir &&
               context_config.device_id == current_config.device_id &&
               context_config.task_exec_config._streams == current_config.task_exec_config._streams &&
               context_config.task_exec_config._threadPreferredCoreType == current_config.task_exec_config._threadPreferredCoreType &&
               context_config.enable_loop_unrolling == current_config.enable_loop_unrolling;
    };

    {
        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::LoadExeNetworkImpl::CreateContext");
        std::lock_guard<std::mutex> lock(engine_mutex);
        if (!canReuseDefaultContext()) {
            m_defaultContext.reset(new CLDNNRemoteCLContext(shared_from_this(), ParamMap(), conf));
        }
    }

    context = m_defaultContext;

    auto transformedNetwork = CloneAndTransformNetwork(network, conf);
    {
        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::LoadExeNetworkImpl::CreateExeNetwork");
        CLDNNExecNetwork::Ptr exeNetwork = std::make_shared<CLDNNExecNetwork>(transformedNetwork, context, conf);
        UpdateStatistics(context);
        return exeNetwork;
    }
}

IExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                                                                const RemoteContext::Ptr &context,
                                                                const std::map<std::string, std::string> &orig_config) {
    InferenceEngine::InputsDataMap _networkInputs = network.getInputsInfo();
    check_inputs(_networkInputs);

    auto casted = std::dynamic_pointer_cast<ClContext>(context);
    if (nullptr == casted) {
        IE_THROW() << "Invalid context";
    }

    CLDNNPlugin::Config conf = getContextImpl(casted)->GetConfig();
    auto config = ConvertPerfHintsToConfig(orig_config, conf);
    UpdateConfig(conf, network, config);

    auto transformedNetwork = CloneAndTransformNetwork(network, conf);
    return std::make_shared<CLDNNExecNetwork>(transformedNetwork, casted, conf);
}

RemoteContext::Ptr clDNNEngine::CreateContext(const ParamMap& params) {
    // parameter map is non-empty
    std::string contextTypeStr = _StrFromParams(params, GPU_PARAM_KEY(CONTEXT_TYPE));

    if (GPU_PARAM_VALUE(OCL) == contextTypeStr) {
        return std::make_shared<CLDNNRemoteCLContext>(shared_from_this(), params, _impl->m_configs.GetDefaultDeviceConfig());
    } else if (GPU_PARAM_VALUE(VA_SHARED) == contextTypeStr) {
#ifdef _WIN32
        return std::make_shared<CLDNNRemoteD3DContext>(shared_from_this(), params, _impl->m_configs.GetDefaultDeviceConfig());
#else
        return std::make_shared<CLDNNRemoteVAContext>(shared_from_this(), params, _impl->m_configs.GetDefaultDeviceConfig());
#endif
    } else {
        IE_THROW() << "Invalid remote context type" << contextTypeStr;
    }
}

RemoteContext::Ptr clDNNEngine::GetDefaultContext(const ParamMap& params) {
    if (nullptr == m_defaultContext) {
        m_defaultContext.reset(new CLDNNRemoteCLContext(shared_from_this(), params, _impl->m_configs.GetDefaultDeviceConfig()));
    }
    return m_defaultContext;
}

void clDNNEngine::SetConfig(const std::map<std::string, std::string> &config) {
    streamsSet = (config.find(PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS) != config.end());
    throttlingSet = config.find(GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE) != config.end() ||
                    config.find(CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE) != config.end();
    std::string device_id;
    if (config.find(PluginConfigInternalParams::KEY_CONFIG_DEVICE_ID) != config.end()) {
        device_id = config.at(PluginConfigInternalParams::KEY_CONFIG_DEVICE_ID);
        _impl->m_configs.GetConfig(device_id).UpdateFromMap(config);
    } else {
        device_id = GetDeviceIDFromConfig(config);
        if (!device_id.empty()) {
            _impl->m_configs.SetDefaultDeviceID(device_id);
            _impl->m_configs.GetConfig(device_id).UpdateFromMap(config);
        } else {
            for (auto& conf : _impl->m_configs) {
                conf.second.UpdateFromMap(config);
            }
        }
    }
}

QueryNetworkResult clDNNEngine::QueryNetwork(const CNNNetwork& network,
                                             const std::map<std::string, std::string>& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::QueryNetwork");
    QueryNetworkResult res;
    CLDNNPlugin::Configs confs = _impl->m_configs;
    std::string device_id = GetDeviceIDFromConfig(config);
    CLDNNPlugin::Config conf = confs.GetConfig(device_id);

    UpdateConfig(conf, network, config);

    if (m_defaultContext == nullptr) {
        m_defaultContext.reset(new CLDNNRemoteCLContext(
            std::const_pointer_cast<InferenceEngine::IInferencePlugin>(shared_from_this()),
            ParamMap(), conf));
    }
    Program prog(m_defaultContext->getImpl()->GetEngine(), conf);
    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "CNNetworkImpl representation is not supported anymore";
    }

    std::unordered_set<std::string> originalOpNames;
    auto originalOps = function->get_ops();
    for (auto&& node : originalOps) {
        originalOpNames.emplace(node->get_friendly_name());
    }

    auto clonedNetwork = CloneAndTransformNetwork(network, conf);
    auto ops = clonedNetwork.getFunction()->get_ordered_ops();
    std::unordered_set<std::string> supported;
    std::unordered_set<std::string> unsupported;

    std::unordered_set<std::string> splitNames;
    std::unordered_set<std::string> concatNames;
    std::unordered_set<std::string> constantsNames;
    std::unordered_set<std::string> depLayerNames;

    std::vector<std::shared_ptr<ngraph::Node>> splits;
    std::vector<std::shared_ptr<ngraph::Node>> concats;
    std::vector<std::shared_ptr<ngraph::Node>> constants;
    std::vector<std::shared_ptr<ngraph::Node>> nextLayerDependent;

    auto layerIsSupported = [&](std::shared_ptr<ngraph::Node> node) {
        if (ngraph::is_type<const ngraph::op::v0::DetectionOutput>(node) ||
            ngraph::is_type<const ngraph::op::v0::PriorBox>(node) ||
            ngraph::is_type<const ngraph::op::v0::PriorBoxClustered>(node) ||
            ngraph::is_type<const ngraph::op::v0::Proposal>(node)) {
            return false;
        } else if (ngraph::is_type<const ngraph::op::v1::Split>(node)) {
            splitNames.emplace(node->get_friendly_name());
            splits.push_back(node);
            return false;
        } else if (ngraph::is_type<const ngraph::op::v0::Concat>(node)) {
            concatNames.emplace(node->get_friendly_name());
            concats.push_back(node);
            return false;
        } else if (ngraph::is_type<const ngraph::op::v1::Reshape>(node) ||
                   ngraph::is_type<const ngraph::op::v0::Squeeze>(node) ||
                   ngraph::is_type<const ngraph::op::v0::Unsqueeze>(node) ||
                   ngraph::is_type<const ngraph::op::v1::Transpose>(node)) {
            depLayerNames.emplace(node->get_friendly_name());
            nextLayerDependent.push_back(node);
            return false;
        } else if (ngraph::is_type<const ngraph::op::v0::Constant>(node)) {
            constantsNames.emplace(node->get_friendly_name());
            constants.push_back(node);
            return false;
        } else if (prog.IsOpSupported(network, node) &&
                   !ngraph::op::is_parameter(node) &&
                   !ngraph::op::is_output(node)) {
            return true;
        } else {
            return false;
        }
    };

    // Get ops after transformations and check if it's supported
    // Transformations might lead to the situation when single node is merged to multiple operations,
    // so we mark original op as supported only if all nodes that it was merged into are supported
    for (auto&& op : ops) {
        for (auto&& fusedLayerName : ngraph::getFusedNamesVector(op)) {
            if (InferenceEngine::details::contains(originalOpNames, fusedLayerName)) {
                if (layerIsSupported(op)) {
                    supported.emplace(fusedLayerName);
                } else {
                    unsupported.emplace(fusedLayerName);
                }
            }
        }
    }

    for (auto&& layerName : supported) {
        if (InferenceEngine::details::contains(unsupported, layerName)) {
            supported.erase(layerName);
        }
    }
    unsupported.clear();

    // Check set of heuristics to produce more efficient hetero sub-graph. Note: checks order is important.
    // 1. Split is marked as supported when all output ops can be offloaded to GPU
    for (const auto & op : splits) {
        bool is_supported = true;
        for (size_t i = 0; i < op->get_output_size(); i++) {
            auto outTensors = op->get_output_target_inputs(i);
            for (auto& t : outTensors) {
                auto output = t.get_node();
                const auto& name = output->get_friendly_name();
                if (!InferenceEngine::details::contains(supported, name) &&
                    !InferenceEngine::details::contains(depLayerNames, name) &&
                    !InferenceEngine::details::contains(concatNames, name) &&
                    !InferenceEngine::details::contains(splitNames, name)) {
                    is_supported = false;
                    break;
                }
            }
        }
        if (is_supported) {
            supported.emplace(op->get_friendly_name());
        }
    }

    // 2. Concat is marked as supported when all inputs can be offloaded to GPU
    for (const auto& op : concats) {
        bool is_supported = true;
        for (size_t i = 0; i < op->get_input_size(); i++) {
            auto input = op->get_input_node_shared_ptr(i);
            const auto& name = input->get_friendly_name();
            if (!InferenceEngine::details::contains(supported, name) &&
                !InferenceEngine::details::contains(depLayerNames, name) &&
                !InferenceEngine::details::contains(concatNames, name)) {
                is_supported = false;
                break;
            }
        }
        if (is_supported) {
            supported.emplace(op->get_friendly_name());
        }
    }

    // 3. Some layers are marked as supported when all inputs and outputs can be offloaded to GPU
    for (const auto& op : nextLayerDependent) {
        bool is_supported = true;
        // both inputs and output should be GPU to remain on GPU
        for (size_t i = 0; i < op->get_input_size(); i++) {
            auto input = op->get_input_node_shared_ptr(i);
            const auto& name = input->get_friendly_name();
            // All inputs must be supported or be a constant
            if (!InferenceEngine::details::contains(supported, name) && !InferenceEngine::details::contains(constantsNames, name)) {
                is_supported = false;
                break;
            }
        }
        for (size_t i = 0; i < op->get_output_size(); i++) {
            auto outTensors = op->get_output_target_inputs(i);
            for (auto& t : outTensors) {
                auto output = t.get_node();
                const auto& name = output->get_friendly_name();
                if (!InferenceEngine::details::contains(supported, name)) {
                    is_supported = false;
                    break;
                }
            }
        }
        if (is_supported) {
            supported.emplace(op->get_friendly_name());
        }
    }

    // 4. Constants are marked as supported when all outputs can be offloaded to GPU
    for (const auto& op : constants) {
        bool is_supported = true;
        for (size_t i = 0; i < op->get_output_size(); i++) {
            auto outTensors = op->get_output_target_inputs(i);
            for (auto& t : outTensors) {
                auto output = t.get_node();
                const auto& name = output->get_friendly_name();
                if (!InferenceEngine::details::contains(supported, name)) {
                    is_supported = false;
                    break;
                }
            }
        }
        if (is_supported) {
            supported.emplace(op->get_friendly_name());
        }
    }

    // Mark original constants/parameters/results ops as supported for each supported operation
    // since rt_info doesn't contain names of constant that are removed during constant folding
    for (auto&& node : originalOps) {
        if (InferenceEngine::details::contains(supported, node->get_friendly_name())) {
            for (auto&& inputNodeOutput : node->input_values()) {
                if (ngraph::op::is_constant(inputNodeOutput.get_node()) || ngraph::op::is_parameter(inputNodeOutput.get_node())) {
                    supported.emplace(inputNodeOutput.get_node()->get_friendly_name());
                }
            }
            for (auto&& outputs : node->outputs()) {
                for (auto&& outputNodeInput : outputs.get_target_inputs()) {
                    if (ngraph::op::is_output(outputNodeInput.get_node())) {
                        supported.emplace(outputNodeInput.get_node()->get_friendly_name());
                    }
                }
            }
        }

        if (ngraph::op::is_constant(node) || ngraph::op::is_parameter(node)) {
                if (!InferenceEngine::details::contains(supported, node->output(0).get_target_inputs().begin()->get_node()->get_friendly_name())) {
                    supported.erase(node->get_friendly_name());
                }
            } else if (ngraph::op::is_output(node)) {
                if (!InferenceEngine::details::contains(supported, node->input_values().begin()->get_node()->get_friendly_name())) {
                    supported.erase(node->get_friendly_name());
                }
            }
    }

    for (auto&& layerName : supported) {
        res.supportedLayersMap.emplace(layerName, GetName());
    }

    return res;
}

Parameter clDNNEngine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::GetConfig");
    Parameter result;

    std::string device_id;
    if (options.find(PluginConfigParams::KEY_DEVICE_ID) != options.end()) {
        device_id = options.find(PluginConfigParams::KEY_DEVICE_ID)->second.as<std::string>();
    }
    Config config = _impl->m_configs.GetConfig(device_id);

    if (config.key_config_map.find(name) != config.key_config_map.end()) {
        result = config.key_config_map.find(name)->second;
    } else {
        IE_THROW() << "Unsupported config key : " << name;
    }
    return result;
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

static float GetGOPS(cldnn::device_info info, cldnn::data_types dt) {
    auto freqGHz = info.gpu_frequency / 1000.f;
    auto numEUs = info.execution_units_count;
    auto opsPerComputeBlock = 0;
    auto computeBlockIPC = 1.0f;
    switch (dt) {
    case cldnn::data_types::u8:
    case cldnn::data_types::i8: {
        if (info.supports_imad) {
            // fma * simd size
            opsPerComputeBlock = 2 * 32;
        } else {
            // separate mul + add instructions for int8 data type
            opsPerComputeBlock = 2 * 16;
            // mul/add instructions can't be executed in parallel, so we need 2 clocks to execute compute block
            computeBlockIPC = 0.5f;
        }
        break;
    }
    case cldnn::data_types::f16: {
        // fma * simd size
        opsPerComputeBlock = 2 * 16;
        break;
    }
    case cldnn::data_types::f32: {
        // fma * simd size
        opsPerComputeBlock = 2 * 8;
        break;
    }

    default: throw std::runtime_error("GetGOPS: Unsupported precision");
    }

    return freqGHz * opsPerComputeBlock * computeBlockIPC * numEUs;
}

Parameter clDNNEngine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "clDNNEngine::GetMetric");
    std::string device_id = GetConfig(CONFIG_KEY(DEVICE_ID), options);

    auto iter = device_map.find(device_id);
    auto device_info = iter != device_map.end() ?
        iter->second->get_info() :
        device_map.begin()->second->get_info();

    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
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
        metrics.push_back(GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
        metrics.push_back(GPU_METRIC_KEY(UARCH_VERSION));
        metrics.push_back(GPU_METRIC_KEY(EXECUTION_UNITS_COUNT));
        metrics.push_back(GPU_METRIC_KEY(MEMORY_STATISTICS));

        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { };
        for (auto const& dev : device_map)
            availableDevices.push_back(dev.first);
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)) {
        IE_SET_METRIC_RETURN(GPU_DEVICE_TOTAL_MEM_SIZE, device_info.max_global_mem_size);
    } else if (name == METRIC_KEY(DEVICE_TYPE)) {
        auto dev_type = device_info.dev_type == cldnn::device_type::discrete_gpu ? Metrics::DeviceType::discrete : Metrics::DeviceType::integrated;
        IE_SET_METRIC_RETURN(DEVICE_TYPE, dev_type);
    } else if (name == METRIC_KEY(DEVICE_GOPS)) {
        std::map<InferenceEngine::Precision, float> gops;
        gops[InferenceEngine::Precision::I8] = GetGOPS(device_info, cldnn::data_types::i8);
        gops[InferenceEngine::Precision::U8] = GetGOPS(device_info, cldnn::data_types::u8);
        gops[InferenceEngine::Precision::FP16] = GetGOPS(device_info, cldnn::data_types::f16);
        gops[InferenceEngine::Precision::FP32] = GetGOPS(device_info, cldnn::data_types::f32);
        IE_SET_METRIC_RETURN(DEVICE_GOPS, gops);
    } else if (name == GPU_METRIC_KEY(EXECUTION_UNITS_COUNT)) {
        IE_SET_METRIC_RETURN(GPU_EXECUTION_UNITS_COUNT, device_info.execution_units_count);
    } else if (name == GPU_METRIC_KEY(UARCH_VERSION)) {
        std::stringstream s;
        if (device_info.gfx_ver.major == 0 && device_info.gfx_ver.minor == 0 && device_info.gfx_ver.revision == 0) {
            s << "unknown";
        } else {
            s << static_cast<int>(device_info.gfx_ver.major) << "."
              << static_cast<int>(device_info.gfx_ver.minor) << "."
              << static_cast<int>(device_info.gfx_ver.revision);
        }
        IE_SET_METRIC_RETURN(GPU_UARCH_VERSION, s.str());
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        auto deviceName = StringRightTrim(device_info.dev_name, "NEO", false);
        deviceName += std::string(" (") + (device_info.dev_type == cldnn::device_type::discrete_gpu ? "dGPU" : "iGPU") + ")";
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, deviceName);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto opt : _impl->m_configs.GetConfig(device_id).key_config_map)
            configKeys.push_back(opt.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities;

        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(BIN));
        capabilities.push_back(METRIC_VALUE(BATCHED_BLOB));
        if (device_info.supports_fp16)
            capabilities.push_back(METRIC_VALUE(FP16));
        if (device_info.supports_imad || device_info.supports_immad)
            capabilities.push_back(METRIC_VALUE(INT8));
        if (device_info.supports_immad)
            capabilities.push_back(METRIC_VALUE(GPU_HW_MATMUL));

        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 2, 1);
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, range);
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, 2);
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, range);
    } else if (name == GPU_METRIC_KEY(MEMORY_STATISTICS)) {
        std::map<std::string, uint64_t> statistics;
        for (auto const &item : statistics_map) {
            for (auto const &kv : item.second) {
                if (!statistics.count(kv.first)) {
                    statistics[kv.first] = kv.second;
                } else {
                    statistics[kv.first] += kv.second;
                }
            }
        }
        IE_SET_METRIC_RETURN(GPU_MEMORY_STATISTICS, statistics);
    } else {
        IE_THROW() << "Unsupported metric key " << name;
    }
}

};  // namespace CLDNNPlugin

static const Version version = { {2, 1}, CI_BUILD_NUMBER, "clDNNPlugin" };
IE_DEFINE_PLUGIN_CREATE_FUNCTION(CLDNNPlugin::clDNNEngine, version)
