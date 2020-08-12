// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for Main Inference Engine API
 *
 * @file ie_plugin.hpp
 */
#pragma once

#include <ie_iextension.h>
#include <ie_icnn_network.hpp>
#include <ie_icore.hpp>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace InferenceEngine {

/**
 * @brief This class is a main plugin interface
 */
class INFERENCE_ENGINE_API_CLASS(IInferencePlugin)
    : public details::IRelease {
public:
    /**
     * @brief Returns plugin version information
     *
     * @param versionInfo Pointer to version info. Is set by plugin
     */
    virtual void GetVersion(const Version*& versionInfo) noexcept = 0;

    /**
     * @brief Creates an executable network from a network object. User can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param ret Reference to a shared ptr of the returned network interface
     * @param network Network object acquired from Core::ReadNetwork
     * @param config Map of pairs: (config parameter name, config parameter value) relevant only for this load operation
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. InferenceEngine::OK if succeeded
     */
    virtual StatusCode LoadNetwork(IExecutableNetwork::Ptr& ret, const ICNNNetwork& network,
                                   const std::map<std::string, std::string>& config, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Creates an executable network from a previously exported network
     *
     * @param ret Reference to a shared ptr of the returned network interface
     * @param modelFileName Path to the location of the exported file
     * @param config Map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation*
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. InferenceEngine::OK if succeeded
     */
    virtual StatusCode ImportNetwork(IExecutableNetwork::Ptr& ret, const std::string& modelFileName,
                                     const std::map<std::string, std::string>& config, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Registers extension within the plugin
     *
     * @param extension Pointer to already loaded extension
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. InferenceEngine::OK if succeeded
     */
    virtual StatusCode AddExtension(InferenceEngine::IExtensionPtr extension,
                                    InferenceEngine::ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Sets configuration for plugin, acceptable keys can be found in ie_plugin_config.hpp
     *
     * @param config Map of pairs: (config parameter name, config parameter value)
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. InferenceEngine::OK if succeeded
     */
    virtual StatusCode SetConfig(const std::map<std::string, std::string>& config, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Query plugin if it supports specified network with specified configuration
     *
     * @param network Network object to query
     * @param config Map of pairs: (config parameter name, config parameter value)
     * @param res Reference to query network result
     */
    virtual void QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                              QueryNetworkResult& res) const noexcept {
        (void)network;
        (void)config;
        res.rc = InferenceEngine::NOT_IMPLEMENTED;
    }

    /**
     * @brief Sets plugin name
     * @param pluginName Plugin name to set
     */
    virtual void SetName(const std::string& pluginName) noexcept = 0;

    /**
     * @brief Returns plugin name
     * @return Plugin name
     */
    virtual std::string GetName() const noexcept = 0;

    /**
     * @brief Sets pointer to ICore interface
     * @param core Pointer to Core interface
     */
    virtual void SetCore(ICore* core) noexcept = 0;

    /**
     * @brief Gets refernce to ICore interface
     * @return Reference to core interface
     */
    virtual const ICore& GetCore() const = 0;

    /**
     * @brief Gets configuration dedicated to plugin behaviour
     * @param name  - value of config corresponding to config key
     * @param options - configuration details for config
     * @return Value of config corresponding to config key
     */
    virtual Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const = 0;

    /**
     * @brief Gets general runtime metric for dedicated hardware
     * @param name  - metric name to request
     * @param options - configuration details for metric
     * @return Metric value corresponding to metric key
     */
    virtual Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const = 0;

    /**
     * @brief      Creates a remote context instance based on a map of parameters
     * @param[in]  params  The map of parameters
     * @return     A remote context object
     */
    virtual RemoteContext::Ptr CreateContext(const ParamMap& params) = 0;

    /**
     * @brief      Provides a default remote context instance if supported by a plugin
     * @return     The default context.
     */
    virtual RemoteContext::Ptr GetDefaultContext() = 0;

    /**
     * @brief Wraps original method
     * IInferencePlugin::LoadNetwork
     * @param network - a network object acquired from InferenceEngine::Core::ReadNetwork
     * @param config string-string map of config parameters relevant only for this load operation
     * @param context - a pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @return Created Executable Network object
     */
    virtual ExecutableNetwork LoadNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                                          RemoteContext::Ptr context) = 0;

    /**
     * @brief Creates an executable network from an previously exported network using plugin implementation
     *        and removes Inference Engine magic and plugin name
     * @param networkModel Reference to network model output stream
     * @param config A string -> string map of parameters
     * @return An Executable network
     */
    virtual ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                            const std::map<std::string, std::string>& config) = 0;

    /**
     * @brief Creates an executable network from an previously exported network using plugin implementation
     *        and removes Inference Engine magic and plugin name
     * @param networkModel Reference to network model output stream
     * @param context - a pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @param config A string -> string map of parameters
     * @return An Executable network
     */
    virtual ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                            const RemoteContext::Ptr& context,
                                            const std::map<std::string, std::string>& config) = 0;

    /**
     * @brief A default virtual destructor
     */
    ~IInferencePlugin() override;
};

}  // namespace InferenceEngine
