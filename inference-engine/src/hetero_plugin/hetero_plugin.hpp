// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "inference_engine.hpp"
#include "description_buffer.hpp"
#include "ie_icore.hpp"
#include "ie_error.hpp"
#include "cpp_interfaces/impl/ie_plugin_internal.hpp"
#include "cpp/ie_plugin_cpp.hpp"
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <utility>


namespace HeteroPlugin {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    IE_SUPPRESS_DEPRECATED_START
    using Plugins = std::unordered_map<std::string, InferenceEngine::InferencePlugin>;
    IE_SUPPRESS_DEPRECATED_END

    using Configs = std::map<std::string, std::string>;
    using Devices = std::vector<std::string>;

    Engine();

    void GetVersion(const InferenceEngine::Version *&versionInfo) noexcept;

    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::ICore * core, InferenceEngine::ICNNNetwork &network, const Configs &config) override;
    void SetConfig(const Configs &config) override;

    void SetAffinity(InferenceEngine::ICNNNetwork& network, const Configs &config);

    void AddExtension(InferenceEngine::IExtensionPtr extension)override;

    void SetLogCallback(InferenceEngine::IErrorListener &listener) override;

    void QueryNetwork(const InferenceEngine::ICNNNetwork &network,
                      const Configs& config, InferenceEngine::QueryNetworkResult &res) const override;

    InferenceEngine::Parameter GetMetric(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter> & options) const override;

    InferenceEngine::Parameter GetConfig(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter> & options) const override;

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::InferencePlugin GetDevicePlugin(const std::string& device) const;

    static Configs GetSupportedConfig(const Configs& config, const InferenceEngine::InferencePlugin& plugin);

    IE_SUPPRESS_DEPRECATED_END

    Plugins GetDevicePlugins(const std::string& targetFallback);

    Plugins GetDevicePlugins(const std::string& targetFallback) const;

    ExecutableNetwork ImportNetworkImpl(std::istream& heteroModel, const Configs& config) override;

    Plugins                                     _plugins;
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
    InferenceEngine::IErrorListener*            _errorListener = nullptr;
};

struct HeteroLayerColorer {
    explicit HeteroLayerColorer(const std::vector<std::string>& devices);

    void operator() (const CNNLayerPtr layer,
                    ordered_properties &printed_properties,
                    ordered_properties &node_properties);

    std::unordered_map<std::string, std::string> deviceColorMap;
};

}  // namespace HeteroPlugin
