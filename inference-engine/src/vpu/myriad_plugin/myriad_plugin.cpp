// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <inference_engine.hpp>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>

#include <vpu/vpu_plugin_config.hpp>
#include <vpu/parsed_config.hpp>

#include "myriad_plugin.h"

using namespace InferenceEngine;
using namespace vpu::MyriadPlugin;

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(ICNNNetwork &network,
                                                          const std::map<std::string, std::string> &config) {
    if (network.getPrecision() != Precision::FP16 &&
        network.getPrecision() != Precision::FP32) {
        THROW_IE_EXCEPTION << "The plugin does not support networks with " << network.getPrecision() << " format.\n"
                           << "Supported format: FP32 and FP16.";
    }

    InputsDataMap networkInputs;
    OutputsDataMap networkOutputs;

    network.getInputsInfo(networkInputs);
    network.getOutputsInfo(networkOutputs);

    auto specifiedDevice = network.getTargetDevice();
    auto supportedDevice = InferenceEngine::TargetDevice::eMYRIAD;
    if (specifiedDevice != InferenceEngine::TargetDevice::eDefault && specifiedDevice != supportedDevice) {
        THROW_IE_EXCEPTION << "The plugin doesn't support target device: " << getDeviceName(specifiedDevice) << ".\n" <<
                           "Supported target device: " << getDeviceName(supportedDevice);
    }

    for (auto networkInput : networkInputs) {
        auto input_precision = networkInput.second->getInputPrecision();

        if (input_precision != Precision::FP16
            && input_precision != Precision::FP32
            && input_precision != Precision::U8) {
            THROW_IE_EXCEPTION << "Input image format " << input_precision << " is not supported yet.\n"
                               << "Supported formats:F16, FP32 and U8.";
        }
    }

    // override what was set globally for plugin, otherwise - override default config without touching config for plugin
    auto configCopy = _config;
    for (auto &&entry : config) {
        configCopy[entry.first] = entry.second;
    }

    return std::make_shared<ExecutableNetwork>(network, _devicePool, configCopy);
}

void Engine::SetConfig(const std::map<std::string, std::string> &userConfig) {
    MyriadConfig myriadConfig(userConfig);

    for (auto &&entry : userConfig) {
        _config[entry.first] = entry.second;
    }
}

void Engine::QueryNetwork(const ICNNNetwork& network, QueryNetworkResult& res) const {
    QueryNetwork(network, {}, res);
}

void Engine::QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                          QueryNetworkResult& res) const {
    auto layerNames = getSupportedLayers(
        network,
        Platform::MYRIAD_2,
        CompilationConfig(),
        std::make_shared<Logger>("GraphCompiler", LogLevel::None, consoleOutput()));

    res.supportedLayers.insert(layerNames.begin(), layerNames.end());
}

Engine::Engine() {
    MyriadConfig config;
    _config = config.getDefaultConfig();
}

// TODO: ImportNetwork and LoadNetwork handle the config parameter in different ways.
// ImportNetwork gets a config provided by an user. LoadNetwork gets the plugin config and merge it with user's config.
// Need to found a common way to handle configs
IExecutableNetwork::Ptr Engine::ImportNetwork(const std::string &modelFileName, const std::map<std::string, std::string> &config) {
    std::ifstream blobFile(modelFileName, std::ios::binary);

    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << details::as_status << NETWORK_NOT_READ;
    }

    IExecutableNetwork::Ptr executableNetwork;
    // Use config provided by an user ignoring default config
    executableNetwork.reset(new ExecutableNetworkBase<ExecutableNetworkInternal>(
                                std::make_shared<ExecutableNetwork>(modelFileName, _devicePool, config)), [](details::IRelease *p) {p->Release();});

    return executableNetwork;
}

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin({1, 6, CI_BUILD_NUMBER, "myriadPlugin"}, std::make_shared<Engine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}
