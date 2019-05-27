// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_plugin.h"
#include "mkldnn_extension_mngr.h"
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <memory>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNWeightsSharing Engine::weightsSharing;
const SimpleDataHash MKLDNNWeightsSharing::simpleCRC;

InferenceEngine::ExecutableNetworkInternal::Ptr
Engine::LoadExeNetworkImpl(InferenceEngine::ICNNNetwork &network, const std::map<std::string, std::string> &config) {
    auto specifiedDevice = network.getTargetDevice();
    auto supportedDevice = InferenceEngine::TargetDevice::eCPU;
    if (specifiedDevice != InferenceEngine::TargetDevice::eDefault && specifiedDevice != supportedDevice) {
        THROW_IE_EXCEPTION << "The plugin doesn't support target device: " << getDeviceName(specifiedDevice) << ".\n" <<
                           "Supported target device: " << getDeviceName(supportedDevice);
    }

    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs;
    network.getInputsInfo(_networkInputs);
    for (auto ii : _networkInputs) {
        auto input_precision = ii.second->getInputPrecision();
        if (input_precision != InferenceEngine::Precision::FP32 &&
            input_precision != InferenceEngine::Precision::I32 &&
            input_precision != InferenceEngine::Precision::U16 &&
            input_precision != InferenceEngine::Precision::I16 &&
            input_precision != InferenceEngine::Precision::I8 &&
            input_precision != InferenceEngine::Precision::U8) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "Input image format " << input_precision << " is not supported yet...";
        }
    }

    // TODO: handle input precision differently - per input and not one per network...

    // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
    Config conf = engConfig;
    conf.readProperties(config);

    if (conf.enableDynamicBatch) {
        conf.batchLimit = network.getBatchSize();
    }

    return std::make_shared<MKLDNNExecNetwork>(network, conf, extensionManager);
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    // accumulate config parameters on engine level
    engConfig.readProperties(config);

    // Pass config to already loaded network
    // TODO: Clarify the behavior of SetConfig method. Should it pass data to already loaded networks?
    if (_loadedNetwork) {
        // ugly casting. can we avoid it?
        auto exe_network =
                dynamic_cast<ExecutableNetworkBase<ExecutableNetworkInternal>*>(_loadedNetwork.get());
        if (exe_network == nullptr)
            THROW_IE_EXCEPTION << "Cannot get executable network!";
        auto exe_network_impl = dynamic_cast<MKLDNNExecNetwork*>(exe_network->getImpl().get());
        if (exe_network_impl == nullptr)
            THROW_IE_EXCEPTION << "Cannot get implementation of executable network!";

        exe_network_impl->setProperty(config);
    }
}

void Engine::AddExtension(InferenceEngine::IExtensionPtr extension) {
    extensionManager->AddExtension(extension);
}

void Engine::QueryNetwork(const ICNNNetwork& network, QueryNetworkResult& res) const {
    QueryNetwork(network, {}, res);
}

void Engine::QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config, QueryNetworkResult& res) const {
    details::CNNNetworkIterator i(const_cast<ICNNNetwork *>(&network));
    while (i != details::CNNNetworkIterator()) {
        try {
            mkldnn::engine eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0));
            // if we can create and have not thrown exception, then layer is supported
            std::unique_ptr <MKLDNNNode>(MKLDNNNode::CreateNode(*i, eng, extensionManager));
            res.supportedLayers.insert((*i)->name);
        } catch (InferenceEngine::details::InferenceEngineException&) {
        }
        i++;
    }
}

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin*& plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin(
                {{1, 6},
                 CI_BUILD_NUMBER,
                 "MKLDNNPlugin"}, std::make_shared<Engine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}
