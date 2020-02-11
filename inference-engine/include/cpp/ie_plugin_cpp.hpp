// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the Inference Engine plugin C++ API
 *
 * @file ie_plugin_cpp.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "cpp/ie_executable_network.hpp"
#include "details/ie_exception_conversion.hpp"
#include "ie_cnn_network.h"
#include "ie_plugin.hpp"
#include "ie_plugin_ptr.hpp"

namespace InferenceEngine {

/**
 * @deprecated Use InferenceEngine::Core instead.
 * @brief This class is a C++ API wrapper for IInferencePlugin.
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::Core instead. Will be removed in 2020 R2") InferencePlugin {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEnginePluginPtr actual;

public:
    /** @brief A default constructor */
    InferencePlugin() = default;

    /**
     * @brief Constructs a plugin instance from the given pointer.
     *
     * @param pointer Initialized Plugin pointer
     */
    explicit InferencePlugin(const InferenceEnginePluginPtr& pointer): actual(pointer) {}

    IE_SUPPRESS_DEPRECATED_END

    /**
     * @copybrief IInferencePlugin::GetVersion
     *
     * Wraps IInferencePlugin::GetVersion
     */
    const Version* GetVersion() {
        const Version* versionInfo = nullptr;
        IE_SUPPRESS_DEPRECATED_START
        actual->GetVersion(versionInfo);
        IE_SUPPRESS_DEPRECATED_END
        if (versionInfo == nullptr) {
            THROW_IE_EXCEPTION << "Unknown device is used";
        }
        return versionInfo;
    }

    /**
     * @copybrief IInferencePlugin::LoadNetwork(IExecutableNetwork::Ptr&, ICNNNetwork&, const std::map<std::string,
     * std::string> &, ResponseDesc*)
     *
     * Wraps IInferencePlugin::LoadNetwork(IExecutableNetwork::Ptr&, ICNNNetwork&, const std::map<std::string,
     * std::string> &, ResponseDesc*)
     *
     * @param network A network object to load
     * @param config A map of configuration options
     * @return Created Executable Network object
     */
    ExecutableNetwork LoadNetwork(ICNNNetwork& network, const std::map<std::string, std::string>& config) {
        IExecutableNetwork::Ptr ret;
        IE_SUPPRESS_DEPRECATED_START
        CALL_STATUS_FNC(LoadNetwork, ret, network, config);
        IE_SUPPRESS_DEPRECATED_END
        return ExecutableNetwork(ret, actual);
    }

    /**
     * @copybrief IInferencePlugin::LoadNetwork(IExecutableNetwork::Ptr&, ICNNNetwork&, const std::map<std::string,
     * std::string> &, ResponseDesc*)
     *
     * Wraps IInferencePlugin::LoadNetwork(IExecutableNetwork::Ptr&, ICNNNetwork&, const std::map<std::string,
     * std::string> &, ResponseDesc*)
     * @param network A network object to load
     * @param config A map of configuration options
     * @return Created Executable Network object
     */
    ExecutableNetwork LoadNetwork(CNNNetwork network, const std::map<std::string, std::string>& config) {
        IExecutableNetwork::Ptr ret;
        IE_SUPPRESS_DEPRECATED_START
        CALL_STATUS_FNC(LoadNetwork, ret, network, config);
        IE_SUPPRESS_DEPRECATED_END
        if (ret.get() == nullptr) THROW_IE_EXCEPTION << "Internal error: pointer to executable network is null";
        return ExecutableNetwork(ret, actual);
    }

    /**
     * @copybrief IInferencePlugin::AddExtension
     *
     * Wraps IInferencePlugin::AddExtension
     *
     * @param extension Pointer to loaded Extension
     */
    void AddExtension(InferenceEngine::IExtensionPtr extension) {
        IE_SUPPRESS_DEPRECATED_START
        CALL_STATUS_FNC(AddExtension, extension);
        IE_SUPPRESS_DEPRECATED_END
    }

    /**
     * @copybrief IInferencePlugin::SetConfig
     *
     * Wraps IInferencePlugin::SetConfig
     * @param config A configuration map
     */
    void SetConfig(const std::map<std::string, std::string>& config) {
        IE_SUPPRESS_DEPRECATED_START
        CALL_STATUS_FNC(SetConfig, config);
        IE_SUPPRESS_DEPRECATED_END
    }

    /**
     * @copybrief IInferencePlugin::ImportNetwork
     *
     * Wraps IInferencePlugin::ImportNetwork
     * @param modelFileName A path to the imported network
     * @param config A configuration map
     * @return Created Executable Network object
     */
    ExecutableNetwork ImportNetwork(const std::string& modelFileName,
                                    const std::map<std::string, std::string>& config) {
        IExecutableNetwork::Ptr ret;
        IE_SUPPRESS_DEPRECATED_START
        CALL_STATUS_FNC(ImportNetwork, ret, modelFileName, config);
        IE_SUPPRESS_DEPRECATED_END
        return ExecutableNetwork(ret, actual);
    }

    /**
     * @copybrief IInferencePlugin::QueryNetwork(const ICNNNetwork&, const std::map<std::string, std::string> &,
     * QueryNetworkResult&)
     *
     * Wraps IInferencePlugin::QueryNetwork(const ICNNNetwork&, const std::map<std::string, std::string> &,
     * QueryNetworkResult&) const
     *
     * @param network A network object to query
     * @param config A configuration map
     * @param res Query results
     */
    void QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                      QueryNetworkResult& res) const {
        IE_SUPPRESS_DEPRECATED_START
        actual->QueryNetwork(network, config, res);
        IE_SUPPRESS_DEPRECATED_END
        if (res.rc != OK) THROW_IE_EXCEPTION << res.resp.msg;
    }

    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief Converts InferenceEngine to InferenceEnginePluginPtr pointer
     *
     * @return Wrapped object
     */
    operator InferenceEngine::InferenceEnginePluginPtr() {
        return actual;
    }

    /**
     * @brief Shared pointer on InferencePlugin object
     *
     */
    using Ptr = std::shared_ptr<InferencePlugin>;

    IE_SUPPRESS_DEPRECATED_END
};
}  // namespace InferenceEngine
