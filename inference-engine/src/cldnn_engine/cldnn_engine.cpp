// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <algorithm>

#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <cmath>
#include <tuple>
#include <cctype>

#include "ie_metric_helpers.hpp"
#include <ie_data.h>
#include <cpp/ie_cnn_network.h>
#include <description_buffer.hpp>
#include <memory>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include "ie_plugin.hpp"
#include "ie_plugin_config.hpp"
#include "details/caseless.hpp"
#include <details/ie_cnn_network_tools.h>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/op/fused/gelu.hpp>
#include <generic_ie.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp>
#include "convert_function_to_cnn_network.hpp"

#undef min
#undef max

#include "cldnn_engine.h"
#include "cldnn_executable_network.h"
#include "cldnn_custom_layer.h"

#ifdef __linux__
#include <dlfcn.h>
#endif

using InferenceEngine::DescriptionBuffer;
using InferenceEngine::TBlob;
using InferenceEngine::Blob;
using namespace InferenceEngine;
using namespace InferenceEngine::gpu;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

struct clDNNEngine::impl {
    CLDNNPlugin::Config m_config;
};

cldnn::device_info clDNNEngine::GetDeviceInfo(const std::map<std::string, std::string> &config) const {
    auto device_info = device_map.begin()->second.get_info();
    if (config.find(PluginConfigParams::KEY_DEVICE_ID) != config.end()) {
        auto val = config.at(PluginConfigParams::KEY_DEVICE_ID);
        if (device_map.find(val) == device_map.end()) {
            THROW_IE_EXCEPTION << "Invalid device ID: " << val;
        }
        device_info = device_map.at(val).get_info();
    }

    return device_info;
}

InferenceEngine::ICNNNetwork::Ptr clDNNEngine::CloneNetwork(const InferenceEngine::ICNNNetwork& network) const {
    std::shared_ptr<ICNNNetwork> clonedNetwork(nullptr);
    if (network.getFunction()) {
        const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            return std::dynamic_pointer_cast<const ::ngraph::opset2::Gelu>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset3::ShuffleChannels>(node);
        };
        CNNNetwork net(network.getFunction());
        auto nGraphFunc = net.getFunction();
        // Disable shape inference (WA for generic operations)
        ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

        // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
        ngraph::pass::CommonOptimizations(transformations_callback).run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet3ToOpSet2(transformations_callback).run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet2ToOpSet1(transformations_callback).run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet1ToLegacy(transformations_callback).run_on_function(nGraphFunc);
        clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, network);
    } else {
        clonedNetwork = cloneNet(network);
    }

    auto implNetwork = std::dynamic_pointer_cast<InferenceEngine::details::CNNNetworkImpl>(clonedNetwork);
    if (implNetwork) {
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        ConstTransformer transformator(implNetwork.get());
        transformator.fullTrim();
    }

    return clonedNetwork;
}

clDNNEngine::clDNNEngine() : m_defaultContext(nullptr) {
    _pluginName = "GPU";
    _impl = std::make_shared<impl>();

    // try loading clDNN engine and get info from it
    {
        cldnn::device_query device_query;
        device_map = device_query.get_available_devices();
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
    CLDNNCustomLayer::LoadFromFile(config_path, _impl->m_config.customLayers, true);
}

auto check_inputs = [](InferenceEngine::InputsDataMap _networkInputs) {
    for (auto ii : _networkInputs) {
        auto input_precision = ii.second->getTensorDesc().getPrecision();
        if (input_precision != InferenceEngine::Precision::FP16 && input_precision != InferenceEngine::Precision::I16
            && input_precision != InferenceEngine::Precision::FP32 && input_precision != InferenceEngine::Precision::U8
            && input_precision != InferenceEngine::Precision::I32 && input_precision != InferenceEngine::Precision::BOOL) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                << "Input image format " << input_precision << " is not supported yet...";
        }
    }
};

ExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork &network,
                                                               const std::map<std::string, std::string> &config) {
    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs;
    network.getInputsInfo(_networkInputs);
    check_inputs(_networkInputs);

    CLDNNPlugin::Config conf = _impl->m_config;
    auto device_info = GetDeviceInfo(config);
    conf.enableInt8 = device_info.supports_imad || device_info.supports_immad;
    conf.UpdateFromMap(config);

    if (conf.enableDynamicBatch) {
        conf.max_dynamic_batch = static_cast<int>(network.getBatchSize());
    }

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
               context_config.device_id == current_config.device_id;
    };

    {
        std::lock_guard<std::mutex> lock(engine_mutex);
        if (!canReuseDefaultContext()) {
            m_defaultContext.reset(new CLDNNRemoteCLContext(shared_from_this(), ParamMap(), conf));
        }
    }

    context = m_defaultContext;

    return std::make_shared<CLDNNExecNetwork>(*CloneNetwork(network), context, conf);
}

ExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork &network,
                                                               RemoteContext::Ptr context,
                                                               const std::map<std::string, std::string> &config) {
    InferenceEngine::InputsDataMap _networkInputs;
    network.getInputsInfo(_networkInputs);
    check_inputs(_networkInputs);

    auto casted = std::dynamic_pointer_cast<ClContext>(context);
    if (nullptr == casted) {
        THROW_IE_EXCEPTION << "Invalid context";
    }

    CLDNNPlugin::Config conf = getContextImpl(casted)->GetConfig();
    auto device_info = GetDeviceInfo(config);
    conf.enableInt8 = device_info.supports_imad || device_info.supports_immad;
    conf.UpdateFromMap(config);

    if (conf.enableDynamicBatch) {
        conf.max_dynamic_batch = static_cast<int>(network.getBatchSize());
    }

    return std::make_shared<CLDNNExecNetwork>(*CloneNetwork(network), casted, conf);
}

RemoteContext::Ptr clDNNEngine::CreateContext(const ParamMap& params) {
    // parameter map is non-empty
    std::string contextTypeStr = _StrFromParams(params, GPU_PARAM_KEY(CONTEXT_TYPE));

    if (GPU_PARAM_VALUE(OCL) == contextTypeStr) {
        auto context = std::make_shared<CLDNNRemoteCLContext>(shared_from_this(), params, _impl->m_config);
        return std::dynamic_pointer_cast<RemoteContext>(context);
    } else if (GPU_PARAM_VALUE(VA_SHARED) == contextTypeStr) {
        #ifdef WIN32
        auto context = std::make_shared<CLDNNRemoteD3DContext>(shared_from_this(), params, _impl->m_config);
        #else
        auto context = std::make_shared<CLDNNRemoteVAContext>(shared_from_this(), params, _impl->m_config);
        #endif
        return std::dynamic_pointer_cast<RemoteContext>(context);
    } else {
        THROW_IE_EXCEPTION << "Invalid remote context type" << contextTypeStr;
    }
}

RemoteContext::Ptr clDNNEngine::GetDefaultContext() {
    if (nullptr == m_defaultContext) {
        m_defaultContext.reset(new CLDNNRemoteCLContext(shared_from_this(), ParamMap(), _impl->m_config));
    }
    return std::dynamic_pointer_cast<RemoteContext>(m_defaultContext);
}

void clDNNEngine::SetConfig(const std::map<std::string, std::string> &config) {
    _impl->m_config.UpdateFromMap(config);
}

void clDNNEngine::QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config, QueryNetworkResult& res) const {
    std::vector <CNNLayer::Ptr> concats;
    std::vector <CNNLayer::Ptr> nextLayerDependent;

    // Verify device id
    GetDeviceInfo(config);

    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
    for (auto layer : sortedLayers) {
        if (CaselessEq<std::string>()(layer->type, "DetectionOutput")) {
        } else if (CaselessEq<std::string>()(layer->type, "PriorBox")) {
        } else if (CaselessEq<std::string>()(layer->type, "Proposal")) {
        } else if (CaselessEq<std::string>()(layer->type, "SimplerNMS")) {
        } else if (CaselessEq<std::string>()(layer->type, "Concat")) {
            concats.push_back(layer);
        } else if (CaselessEq<std::string>()(layer->type, "reshape")) {
            nextLayerDependent.push_back(layer);
        } else if (CaselessEq<std::string>()(layer->type, "permute")) {
            nextLayerDependent.push_back(layer);
        } else if (CaselessEq<std::string>()(layer->type, "Const")) {
            nextLayerDependent.push_back(layer);
        } else if (CLDNNGraph::IsLayerSupported(layer->type)) {
            res.supportedLayersMap.insert({ layer->name, GetName() });
        }
    }

    // evaluation of concats - if all parent layers are supported, only in this case we
    // will mark concat as a supported for GPU
    for (const auto &concat : concats) {
        // take all parrents.
        bool supported = true;
        for (DataWeakPtr insData : concat->insData) {
            CNNLayerPtr prev = insData.lock()->getCreatorLayer().lock();
            // verify if previous layer is not supported or if it in the list of not defined layers yet
            // not defined layers are treated as layers which will be assigned to GPU if next layer is assigned to GPU
            if (res.supportedLayersMap.find(prev->name) == res.supportedLayersMap.end()
                && std::find(nextLayerDependent.begin(), nextLayerDependent.end(), prev) == nextLayerDependent.end()) {
                supported = false;
            }
        }
        if (supported) {
            res.supportedLayersMap.insert({ concat->name, GetName() });
        }
    }

    // evaluation of constant blobs - if all consumers are on GPU,
    // then leave it on GPU, else - move to other device
    for (auto cnl = nextLayerDependent.rbegin();
        cnl != nextLayerDependent.rend();
        cnl++) {
        bool supported = true;
        for (DataPtr out : (*cnl)->outData) {
            for (auto ol : out->getInputTo()) {
                if (res.supportedLayersMap.find(ol.second->name) == res.supportedLayersMap.end()) {
                    supported = false;
                }
            }
        }
        std::cout << (*cnl)->name << " is " << (supported ? "GPU" : "CPU") << std::endl;

        if (supported) {
            res.supportedLayersMap.insert({ (*cnl)->name, GetName() });
        }
    }
}

Parameter clDNNEngine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    Parameter result;
    auto option = _impl->m_config.key_config_map.find(name);
    if (option != _impl->m_config.key_config_map.end()) {
        result = option->second;
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key : " << name;
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

Parameter clDNNEngine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
    auto device_id = GetConfig(CONFIG_KEY(DEVICE_ID), {});
    if (options.find(CONFIG_KEY(DEVICE_ID)) != options.end())
        device_id = options.at(CONFIG_KEY(DEVICE_ID)).as<std::string>();

    auto iter = device_map.find(device_id);
    auto device_info = iter != device_map.end() ?
        iter->second.get_info() :
        device_map.begin()->second.get_info();

    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(AVAILABLE_DEVICES));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_STREAMS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { };
        for (auto const& dev : device_map)
            availableDevices.push_back(dev.first);
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, StringRightTrim(device_info.dev_name, "NEO", false));
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto opt : _impl->m_config.key_config_map)
            configKeys.push_back(opt.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities;

        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(BIN));
        if (device_info.supports_fp16)
            capabilities.push_back(METRIC_VALUE(FP16));
        if (device_info.supports_imad || device_info.supports_immad)
            capabilities.push_back(METRIC_VALUE(INT8));

        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 2, 1);
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, range);
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, 2);
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, range);
    } else {
        THROW_IE_EXCEPTION << "Unsupported metric key " << name;
    }
}

};  // namespace CLDNNPlugin

IE_SUPPRESS_DEPRECATED_START

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin*& plugin, ResponseDesc* resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin(
            { 2, 1,
             CI_BUILD_NUMBER,
             "clDNNPlugin" }, std::make_shared<CLDNNPlugin::clDNNEngine>());
        return OK;
    }
    catch (std::exception & ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

