// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the Inference Engine plugin C++ API
 * @file ie_plugin_cpp.hpp
 */
#pragma once

#include <map>
#include <string>
#include <memory>

#include "ie_plugin.hpp"
#include "details/ie_exception_conversion.hpp"
#include "cpp/ie_executable_network.hpp"
#include "ie_plugin_ptr.hpp"
#include "ie_cnn_network.h"


namespace InferenceEngine {

/**
 * @brief This class is a C++ API wrapper for IInferencePlugin.
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class InferencePlugin {
    InferenceEnginePluginPtr actual;

public:
    /** @brief A default constructor */
    InferencePlugin() = default;

    /**
     * @brief Constructs a plugin instance from the given pointer.
     * @param pointer Initialized Plugin pointer
     */
    explicit InferencePlugin(const InferenceEnginePluginPtr &pointer) : actual(pointer) {}

    /**
     * @brief Wraps original method
     * IInferencePlugin::GetVersion
     */
    const Version *GetVersion() {
        const Version *versionInfo = nullptr;
        actual->GetVersion(versionInfo);
        if (versionInfo == nullptr) {
            THROW_IE_EXCEPTION << "Unknown device is used";
        }
        return versionInfo;
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::LoadNetwork(IExecutableNetwork::Ptr&, ICNNNetwork&, const std::map<std::string, std::string> &, ResponseDesc*).
     * @param network A network object to load
     * @param config A map of configuration options
     * @return Created Executable Network object
     */
    ExecutableNetwork LoadNetwork(ICNNNetwork &network, const std::map<std::string, std::string> &config) {
        IExecutableNetwork::Ptr ret;
        CALL_STATUS_FNC(LoadNetwork, ret, network, config);
        return ExecutableNetwork(ret, actual);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::LoadNetwork(IExecutableNetwork::Ptr&, ICNNNetwork&, const std::map<std::string, std::string> &, ResponseDesc*).
     * @param network A network object to load
     * @param config A map of configuration options
     * @return Created Executable Network object
     */
    ExecutableNetwork LoadNetwork(CNNNetwork network, const std::map<std::string, std::string> &config) {
        IExecutableNetwork::Ptr ret;
        CALL_STATUS_FNC(LoadNetwork, ret, network, config);
        if (ret.get() == nullptr) THROW_IE_EXCEPTION << "Internal error: pointer to executable network is null";
        return ExecutableNetwork(ret, actual);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::AddExtension
     * @param extension Pointer to loaded Extension
     */
    void AddExtension(InferenceEngine::IExtensionPtr extension) {
        CALL_STATUS_FNC(AddExtension, extension);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::SetConfig
     * @param config A configuration map
     */
    void SetConfig(const std::map<std::string, std::string> &config) {
        CALL_STATUS_FNC(SetConfig, config);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::ImportNetwork
     * @param modelFileName A path to the imported network
     * @param config A configuration map
     * @return Created Executable Network object
     */
    ExecutableNetwork ImportNetwork(const std::string &modelFileName, const std::map<std::string, std::string> &config) {
        IExecutableNetwork::Ptr ret;
        CALL_STATUS_FNC(ImportNetwork, ret, modelFileName, config);
        return ExecutableNetwork(ret, actual);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::QueryNetwork(const ICNNNetwork&, const std::map<std::string, std::string> &, QueryNetworkResult&) const
     * @param network A network object to query
     * @param config A configuration map
     * @param res Query results
     */
    void QueryNetwork(const ICNNNetwork &network, const std::map<std::string, std::string> &config, QueryNetworkResult &res) const {
        actual->QueryNetwork(network, config, res);
        if (res.rc != OK) THROW_IE_EXCEPTION << res.resp.msg;
    }

    /**
     * @brief Converts InferenceEngine to InferenceEnginePluginPtr pointer
     * @return Wrapped object
     */
    operator InferenceEngine::InferenceEnginePluginPtr() {
        return actual;
    }

    /**
     * @brief Shared pointer on InferencePlugin object
     */
    using Ptr = std::shared_ptr<InferencePlugin>;
};
}  // namespace InferenceEngine
