// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "inference_engine.hpp"
#include "hetero_plugin_base.hpp"
#include "ie_ihetero_plugin.hpp"
#include "description_buffer.hpp"
#include "ie_icore.hpp"
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
    LoadExeNetworkImpl(const InferenceEngine::ICore * core, InferenceEngine::ICNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;
    void SetConfig(const std::map<std::string, std::string> &config) override;

    IE_SUPPRESS_DEPRECATED_START
    void SetDeviceLoader(const std::string &device, InferenceEngine::IHeteroDeviceLoader::Ptr pLoader);
    IE_SUPPRESS_DEPRECATED_END

    void SetAffinity(InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string> &config);

    void AddExtension(InferenceEngine::IExtensionPtr extension)override;

    void SetLogCallback(InferenceEngine::IErrorListener &listener) override;

    void QueryNetwork(const InferenceEngine::ICNNNetwork &network,
        const std::map<std::string, std::string>& config, InferenceEngine::QueryNetworkResult &res) const override;

    InferenceEngine::Parameter GetMetric(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter> & options) const override;

    InferenceEngine::Parameter GetConfig(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter> & options) const override;

private:
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
    InferenceEngine::MapDeviceLoaders _deviceLoaders;
    InferenceEngine::IErrorListener* error_listener = nullptr;
};

INFERENCE_ENGINE_API(InferenceEngine::StatusCode) CreateHeteroPluginEngine(
        InferenceEngine::IInferencePlugin *&plugin,
        InferenceEngine::ResponseDesc *resp) noexcept;

}  // namespace HeteroPlugin
