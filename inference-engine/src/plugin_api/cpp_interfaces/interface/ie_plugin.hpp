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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "details/ie_no_copy.hpp"
#include "ie_api.h"
#include "ie_core.hpp"
#include "ie_iexecutable_network.hpp"
#include "ie_version.hpp"

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
     * @brief A default virtual destructor
     */
    ~IInferencePlugin() override;
};

/**
 * @brief Creates the default instance of the interface (per plugin)
 *
 * @param plugin Pointer to the plugin
 * @param resp Pointer to the response message that holds a description of an error if any occurred
 * @return Status code of the operation. InferenceEngine::OK if succeeded
 */
INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin*& plugin, ResponseDesc* resp) noexcept;

}  // namespace InferenceEngine
