// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "inference_engine.hpp"
#include "ie_ihetero_plugin.hpp"
#include "description_buffer.hpp"
#include "ie_error.hpp"
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

namespace HeteroPlugin {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    Engine();

    void GetVersion(const InferenceEngine::Version *&versionInfo) noexcept;

    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(InferenceEngine::ICNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;
    void SetConfig(const std::map<std::string, std::string> &config) override;

    void SetDeviceLoader(const std::string &device, InferenceEngine::IHeteroDeviceLoader::Ptr pLoader);

    void SetAffinity(InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string> &config);

    void AddExtension(InferenceEngine::IExtensionPtr extension)override;

    void SetLogCallback(InferenceEngine::IErrorListener &listener) override;
private:
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
    InferenceEngine::MapDeviceLoaders _deviceLoaders;
    InferenceEngine::IErrorListener* error_listener = nullptr;
};

}  // namespace HeteroPlugin
