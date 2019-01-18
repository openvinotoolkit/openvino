// Copyright (C) 2018 Intel Corporation
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
     */
    explicit InferencePlugin(const InferenceEnginePluginPtr &pointer) : actual(pointer) {}

    /**
     * @brief Wraps original method
     * IInferencePlugin::GetVersion
     */
    const Version *GetVersion() {
        const Version *versionInfo = nullptr;
        actual->GetVersion(versionInfo);
        return versionInfo;
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::LoadNetwork
     */
    void LoadNetwork(ICNNNetwork &network) {
        CALL_STATUS_FNC(LoadNetwork, network);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::LoadNetwork(IExecutableNetwork::Ptr&, ICNNNetwork&, const std::map<std::string, std::string> &, ResponseDesc*).
     */
    ExecutableNetwork LoadNetwork(ICNNNetwork &network, const std::map<std::string, std::string> &config) {
        IExecutableNetwork::Ptr ret;
        CALL_STATUS_FNC(LoadNetwork, ret, network, config);
        return ExecutableNetwork(ret);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::LoadNetwork(IExecutableNetwork::Ptr&, ICNNNetwork&, const std::map<std::string, std::string> &, ResponseDesc*).
     */
    ExecutableNetwork LoadNetwork(CNNNetwork network, const std::map<std::string, std::string> &config) {
        IExecutableNetwork::Ptr ret;
        CALL_STATUS_FNC(LoadNetwork, ret, network, config);
        if (ret.get() == nullptr) THROW_IE_EXCEPTION << "Internal error: pointer to executable network is null";
        return ExecutableNetwork(ret);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::Infer(const BlobMap&, BlobMap&, ResponseDesc *resp)
     */
    void Infer(const BlobMap &input, BlobMap &result) {
        CALL_STATUS_FNC(Infer, input, result);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::GetPerformanceCounts
     */
    std::map<std::string, InferenceEngineProfileInfo> GetPerformanceCounts() const {
        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        CALL_STATUS_FNC(GetPerformanceCounts, perfMap);
        return perfMap;
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::AddExtension
     */
    void AddExtension(InferenceEngine::IExtensionPtr extension) {
        CALL_STATUS_FNC(AddExtension, extension);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::SetConfig
     */
    void SetConfig(const std::map<std::string, std::string> &config) {
        CALL_STATUS_FNC(SetConfig, config);
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::ImportNetwork
    */
    ExecutableNetwork  ImportNetwork(const std::string &modelFileName, const std::map<std::string, std::string> &config) {
        IExecutableNetwork::Ptr ret;
        CALL_STATUS_FNC(ImportNetwork, ret, modelFileName, config);
        return ExecutableNetwork(ret);
    }

    /**
     * @depricated Use the version with config parameter
     * @brief Wraps original method
     * IInferencePlugin::QueryNetwork
     */
    void QueryNetwork(const ICNNNetwork &network, QueryNetworkResult &res) const {
        actual->QueryNetwork(network, res);
        if (res.rc != OK) THROW_IE_EXCEPTION << res.resp.msg;
    }

    /**
     * @brief Wraps original method
     * IInferencePlugin::QueryNetwork
     */
    void QueryNetwork(const ICNNNetwork &network, const std::map<std::string, std::string> &config, QueryNetworkResult &res) const {
        actual->QueryNetwork(network, config, res);
        if (res.rc != OK) THROW_IE_EXCEPTION << res.resp.msg;
    }

    /**
     * @brief Returns wrapped object
     */
    operator InferenceEngine::InferenceEnginePluginPtr() {
        return actual;
    }

    /**
    * @return wrapped Hetero object if underlined object is HeteroPlugin instance, nullptr otherwise
    */
    operator InferenceEngine::HeteroPluginPtr() {
        return actual;
    }

    /**
     * @brief Shared pointer on InferencePlugin object
     */
    using Ptr = std::shared_ptr<InferencePlugin>;
};
}  // namespace InferenceEngine
