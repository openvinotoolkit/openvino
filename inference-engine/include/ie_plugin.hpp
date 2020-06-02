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
#include "ie_error.hpp"
#include "ie_iexecutable_network.hpp"
#include "ie_version.hpp"

/**
 * @def INFERENCE_PLUGIN_API(type)
 * @brief Defines Inference Engine Plugin API method
 * @param type A plugin type
 */

#if defined(_WIN32)
#ifdef IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#define INFERENCE_PLUGIN_API(type) extern "C" __declspec(dllexport) type
#else
#define INFERENCE_PLUGIN_API(type) extern "C" type
#endif
#elif (__GNUC__ >= 4)  // NOLINT
#ifdef IMPLEMENT_INFERENCE_ENGINE_PLUGIN
#define INFERENCE_PLUGIN_API(type) extern "C" __attribute__((visibility("default"))) type
#else
#define INFERENCE_PLUGIN_API(type) extern "C" type
#endif
#else
#define INFERENCE_PLUGIN_API(TYPE) extern "C" TYPE
#endif

namespace InferenceEngine {

/**
 * @brief Responce structure encapsulating information about supported layer
 */
struct QueryNetworkResult {
    /**
     * @brief A map of supported layers:
     * - key - a layer name
     * - value - a device name on which layer is assigned
     */
    std::map<std::string, std::string> supportedLayersMap;

    /**
     * @brief A status code
     */
    StatusCode rc = OK;

    /**
     * @brief Response mssage
     */
    ResponseDesc resp;
};

/**
 * @deprecated Use InferenceEngine::Core instead. Will be removed in 2021.1
 * @brief This class is a main plugin interface
 */
class INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::Core instead. Will be removed in 2021.1")
    INFERENCE_ENGINE_API_CLASS(IInferencePlugin)
    : public details::IRelease {
public:
    /**
     * @brief Returns plugin version information
     *
     * @param versionInfo Pointer to version info. Is set by plugin
     */
    virtual void GetVersion(const Version*& versionInfo) noexcept = 0;

    /**
     * @deprecated IErrorListener is not used anymore. StatusCode is provided in case of unexpected situations
     * This API will be removed in 2021.1 release.
     * @brief Sets logging callback
     *
     * Logging is used to track what is going on inside
     * @param listener Logging sink
     */
    IE_SUPPRESS_DEPRECATED_START
    INFERENCE_ENGINE_DEPRECATED("IErrorListener is not used anymore. StatusCode is provided in case of unexpected situations")
    virtual void SetLogCallback(IErrorListener& listener) noexcept = 0;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Creates an executable network from a network object. User can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param ret Reference to a shared ptr of the returned network interface
     * @param network Network object acquired from CNNNetReader
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
IE_SUPPRESS_DEPRECATED_START
INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin*& plugin, ResponseDesc* resp) noexcept;
IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
