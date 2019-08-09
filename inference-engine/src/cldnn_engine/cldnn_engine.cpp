// Copyright (C) 2018-2019 Intel Corporation
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

#include "ie_metric_helpers.hpp"
#include <debug.h>
#include <ie_data.h>
#include <cpp/ie_cnn_network.h>
#include <description_buffer.hpp>
#include <memory>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include "ie_plugin.hpp"
#include "ie_plugin_config.hpp"
#include "details/caseless.hpp"
#include <details/ie_cnn_network_tools.h>

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
using namespace details;

namespace CLDNNPlugin {

struct clDNNEngine::impl {
    CLDNNPlugin::Config m_config;
};

clDNNEngine::clDNNEngine() {
    _pluginName = "GPU";
    _impl = std::make_shared<impl>();

    // try loading clDNN engine and get info from it
    {
        cldnn::engine info_engine(cldnn::engine_configuration(
            false,
            false,
            false,
            std::string(),
            std::string(),
            true,
            std::string(),
            std::string(),
            cldnn::priority_mode_types::disabled,
            cldnn::throttle_mode_types::disabled,
            true,
            1));

        engine_info = info_engine.get_info();
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

ExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::ICore * /*core*/, InferenceEngine::ICNNNetwork &network,
                                                               const std::map<std::string, std::string> &config) {
    IE_SUPPRESS_DEPRECATED_START
    auto specifiedDevice = network.getTargetDevice();
    auto supportedDevice = InferenceEngine::TargetDevice::eGPU;
    if (specifiedDevice != InferenceEngine::TargetDevice::eDefault && specifiedDevice != supportedDevice) {
        THROW_IE_EXCEPTION << "The plugin doesn't support target device: " << getDeviceName(specifiedDevice) << ".\n" <<
                           "Supported target device: " << getDeviceName(supportedDevice);
    }
    IE_SUPPRESS_DEPRECATED_END

    CLDNNPlugin::Config conf = this->_impl->m_config;
    conf.UpdateFromMap(config);

    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs;
    network.getInputsInfo(_networkInputs);
    for (auto ii : _networkInputs) {
        auto input_precision = ii.second->getTensorDesc().getPrecision();
        if (input_precision != InferenceEngine::Precision::FP16 && input_precision != InferenceEngine::Precision::I16
            && input_precision != InferenceEngine::Precision::FP32 && input_precision != InferenceEngine::Precision::U8
            && input_precision != InferenceEngine::Precision::I32) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "Input image format " << input_precision << " is not supported yet...";
        }
    }

    if (conf.enableDynamicBatch) {
        conf.max_dynamic_batch = static_cast<int>(network.getBatchSize());
    }

    return std::make_shared<CLDNNExecNetwork>(network, conf);
}

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin(
                {2, 0,
                 CI_BUILD_NUMBER,
                 "clDNNPlugin"}, std::make_shared<clDNNEngine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

void clDNNEngine::SetConfig(const std::map<std::string, std::string> &config) {
    _impl->m_config.UpdateFromMap(config);
}

void clDNNEngine::QueryNetwork(const ICNNNetwork& network, QueryNetworkResult& res) const {
    QueryNetwork(network, {}, res);
}

void clDNNEngine::QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config, QueryNetworkResult& res) const {
    std::vector <CNNLayer::Ptr> concats;
    std::vector <CNNLayer::Ptr> nextLayerDependent;

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
            IE_SUPPRESS_DEPRECATED_START
            res.supportedLayers.insert(layer->name);
            IE_SUPPRESS_DEPRECATED_END
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
            IE_SUPPRESS_DEPRECATED_START
            res.supportedLayers.insert(concat->name);
            IE_SUPPRESS_DEPRECATED_END
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
            IE_SUPPRESS_DEPRECATED_START
            res.supportedLayers.insert((*cnl)->name);
            IE_SUPPRESS_DEPRECATED_END
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

Parameter clDNNEngine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(AVAILABLE_DEVICES));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS));
        metrics.push_back(METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_STREAMS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { "" };
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, std::string(engine_info.ocl_device_name));
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto opt : _impl->m_config.key_config_map)
            configKeys.push_back(opt.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities;

        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(BIN));
        if (engine_info.supports_fp16)
            capabilities.push_back(METRIC_VALUE(FP16));
        if (engine_info.supports_imad || engine_info.supports_immad)
            capabilities.push_back(METRIC_VALUE(INT8));

        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (name == METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(NUMBER_OF_WAITING_INFER_REQUESTS, CLDNNExecNetwork::GetWaitingCounter());
    } else if (name == METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(NUMBER_OF_EXEC_INFER_REQUESTS, CLDNNExecNetwork::GetRunningCounter());
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
