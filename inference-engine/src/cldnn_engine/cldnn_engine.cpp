// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <algorithm>

#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <cmath>
#include <debug.h>
#include <ie_data.h>
#include <cpp/ie_cnn_network.h>
#include <description_buffer.hpp>
#include <memory>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include "ie_plugin.hpp"
#include "ie_plugin_config.hpp"
#include "details/caseless.hpp"

#undef min
#undef max

#include "cldnn_engine.h"
#include "cldnn_graph.h"
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
    CLDNNGraph::Config m_config;
};

clDNNEngine::clDNNEngine() {
    _impl = new impl;

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

clDNNEngine::~clDNNEngine() {
    if (_impl) {
        delete _impl;
        _impl = nullptr;
    }
}

ExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(InferenceEngine::ICNNNetwork &network,
                                                               const std::map<std::string, std::string> &config) {
    auto specifiedDevice = network.getTargetDevice();
    auto supportedDevice = InferenceEngine::TargetDevice::eGPU;
    if (specifiedDevice != InferenceEngine::TargetDevice::eDefault && specifiedDevice != supportedDevice) {
        THROW_IE_EXCEPTION << "The plugin doesn't support target device: " << getDeviceName(specifiedDevice) << ".\n" <<
                           "Supported target device: " << getDeviceName(supportedDevice);
    }

    CLDNNGraph::Config conf = this->_impl->m_config;
    conf.LoadFromMap(config);

    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs;
    network.getInputsInfo(_networkInputs);
    for (auto ii : _networkInputs) {
        auto input_precision = ii.second->getInputPrecision();
        if (input_precision != InferenceEngine::Precision::FP16 && input_precision != InferenceEngine::Precision::I16
            && input_precision != InferenceEngine::Precision::FP32 && input_precision != InferenceEngine::Precision::U8) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "Input image format " << input_precision << " is not supported yet...";
        }
    }
    // todo: handle input precision differently - per input and not one per network...

    int max_batch = -1;
    if (conf.enableDynamicBatch) {
        max_batch = network.getBatchSize();
    }

    return std::make_shared<CLDNNGraph>(network, conf, max_batch);
}

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin(
                {1, 4,
#ifdef CLDNN_VERSION
                 CLDNN_VERSION,
#else
                 CI_BUILD_NUMBER,
#endif
                 "clDNNPlugin"}, std::make_shared<clDNNEngine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

void clDNNEngine::SetConfig(const std::map<std::string, std::string> &config) {
    _impl->m_config.LoadFromMap(config);
}

void clDNNEngine::QueryNetwork(const ICNNNetwork& network, QueryNetworkResult& res) const {
    QueryNetwork(network, {}, res);
}

void clDNNEngine::QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config, QueryNetworkResult& res) const {
    details::CNNNetworkIterator i(const_cast<ICNNNetwork *>(&network));

    std::vector <CNNLayer::Ptr> concats;
    std::vector <CNNLayer::Ptr> constantBlobs;

    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;

        if (CaselessEq<std::string>()(layer->type, "DetectionOutput")) {
        } else if (CaselessEq<std::string>()(layer->type, "PriorBox")) {
        } else if (CaselessEq<std::string>()(layer->type, "Proposal")) {
        } else if (CaselessEq<std::string>()(layer->type, "SimplerNMS")) {
        } else if (CaselessEq<std::string>()(layer->type, "Concat")) {
            concats.push_back(layer);
        } else if (CaselessEq<std::string>()(layer->type, "Const")) {
            constantBlobs.push_back(layer);
        } else if (CLDNNGraph::IsLayerSupported(layer->type)) {
            res.supportedLayers.insert((*i)->name);
        }

        i++;
    }

    // evaluation of concats - if all parent layers are supported, only in this case we
    // will mark concat as a supported for GPU
    for (const auto &concat : concats) {
        // take all parrents.
        bool supported = true;
        for (DataWeakPtr insData : concat->insData) {
            CNNLayerPtr prev = insData.lock()->getCreatorLayer().lock();
            if (res.supportedLayers.find(prev->name) == res.supportedLayers.end()) {
                supported = false;
            }
        }
        if (supported)
            res.supportedLayers.insert(concat->name);
    }

    // evaluation of constant blobs - if all consumers are on GPU,
    // then leave it on GPU, else - move to other device
    for (const auto &cblob : constantBlobs) {
        bool supported = true;
        for (DataPtr out : cblob->outData) {
            CNNLayerPtr prev = out->getCreatorLayer().lock();
            if (res.supportedLayers.find(prev->name) == res.supportedLayers.end()) {
                supported = false;
            }
        }
        if (supported)
            res.supportedLayers.insert(cblob->name);
    }
}


};  // namespace CLDNNPlugin
