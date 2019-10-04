// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides interface to register custom hetero functionality
 * @file ie_ihetero_plugin.hpp
 */
#pragma once
#include <map>
#include <string>
#include <memory>
#include <ie_api.h>
#include <ie_icnn_network.hpp>
#include <ie_iexecutable_network.hpp>
#include <ie_plugin.hpp>

namespace InferenceEngine {

/**
 * @deprecated Use InferenceEngine::Core to work with HETERO device
 * @brief This interface describes a mechanism of custom loaders to be used in heterogeneous
 * plugin during setting of affinity and loading of split sub-network to the plugins
 * The custom loader can define addition settings for the plugins or network loading
 * Examples of cases when this interface should be implemented in the application:
 * 
 * 1. add custom layers to existing plugins if it is not pointed to the heterogeneous plugin
 *  or registration of custom layer is different than supported in available public plugins
 * 
 * 2. set affinity manually for the same plugin being initialized by different parameters,
 *  e.g different device id
 *  In this case there will be mapping of
 *    Device1 > HeteroDeviceLoaderImpl1
 *    Device2 > HeteroDeviceLoaderImpl2
 *  the affinity should be pointed manually, the implementation of HeteroDeviceLoaderImpl1 and
 *  HeteroDeviceLoaderImpl2 should be in the application, and these device loaders should be registered
 *  through calling of
 *  IHeteroInferencePlugin::SetDeviceLoader("Device1", HeteroDeviceLoaderImpl1)
 *  IHeteroInferencePlugin::SetDeviceLoader("Device2", HeteroDeviceLoaderImpl2)
*/
class INFERENCE_ENGINE_DEPRECATED INFERENCE_ENGINE_API_CLASS(IHeteroDeviceLoader) {
public:
    virtual ~IHeteroDeviceLoader();

    /**
     * @deprecated Use InferenceEngine::Core with HETERO device in InferenceEngine::Core::LoadNetwork.
     * @brief Loads network to the device. The instantiation of plugin should be in the implementation
     * of the IHeteroDeviceLoader. As well setting of special config option should happen in the
     * implementation as well
     * @param device Loading of network should happen for this device
     * @param ret Reference to a shared ptr of the returned executable network instance
     * @param network Network object acquired from CNNNetReader
     * @param config Map of configuration settings relevant only for current load operation
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    INFERENCE_ENGINE_DEPRECATED
    virtual StatusCode LoadNetwork(
        const std::string& device,
        IExecutableNetwork::Ptr &ret,
        ICNNNetwork &network,
        const std::map<std::string, std::string> &config,
        ResponseDesc *resp) noexcept = 0;

    /**
     * @deprecated Use the IHeteroDeviceLoader::QueryNetwork
     * @brief This function calls plugin function QueryNetwork for the plugin being instantiated
     * in the implementation of IHeteroDeviceLoader
     * @param device QueryNetwork will be executed for this device
     * @param network Network object acquired from CNNNetReader
     * @param res Query network result object
     */
    INFERENCE_ENGINE_DEPRECATED
    virtual void QueryNetwork(const std::string &device,
                              const ICNNNetwork &network,
                              QueryNetworkResult &res) noexcept {
        IE_SUPPRESS_DEPRECATED_START
        QueryNetwork(device, network, { }, res);
        IE_SUPPRESS_DEPRECATED_END
    }

    /**
     * @deprecated Use InferenceEngine::Core with HETERO device in InferenceEngine::Core::QueryNetwork.
     * @brief This function calls plugin function QueryNetwork for the plugin being instantiated
     * in the implementation of IHeteroDeviceLoader
     * @param device QueryNetwork will be executed for this device
     * @param network Network object acquired from CNNNetReader
     * @param config  Network configuration parameters
     * @param res Query network result object
     */
    INFERENCE_ENGINE_DEPRECATED
    virtual void QueryNetwork(const std::string &device,
                              const ICNNNetwork &network,
                              const std::map<std::string, std::string>& config,
                              QueryNetworkResult &res) noexcept = 0;


    /**
     * @deprecated Use InferenceEngine::Core with HETERO device in InferenceEngine::Core::QueryNetwork.
     * @brief Sets log callback
     * @param listener A reference to IErrorListener object
     */
    virtual void SetLogCallback(IErrorListener &listener) = 0;

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief Shared pointer to IHeteroDeviceLoader instance
     */
    using Ptr = std::shared_ptr<IHeteroDeviceLoader>;
    IE_SUPPRESS_DEPRECATED_END
};

IE_SUPPRESS_DEPRECATED_START
/**
 * @brief Represents map from device name to device-specific loader
 */
using MapDeviceLoaders = std::map<std::string, InferenceEngine::IHeteroDeviceLoader::Ptr>;
IE_SUPPRESS_DEPRECATED_END

/**
 * @deprecated Use InferenceEngine::Core with HETERO mode in LoadNetwork, QueryNetwork, etc
 * @brief This interface extends regular plugin interface for heterogeneous case. Not all plugins
 * implements it. The main purpose of this interface - to register loaders and have an ability
 * to get default settings for affinity on certain devices.
 */
class INFERENCE_ENGINE_DEPRECATED INFERENCE_ENGINE_API_CLASS(IHeteroInferencePlugin) : public IInferencePlugin {
public:
    virtual ~IHeteroInferencePlugin();

    /**
     * @deprecated Use InferenceEngine::Core to work with HETERO device
     * Registers device loader for the device
     * @param device - the device name being used in CNNNLayer::affinity
     * @param loader - helper class allowing to analyze if layers are supported and allow
     * to load network to the plugin being defined in the IHeteroDeviceLoader implementation
     */
    IE_SUPPRESS_DEPRECATED_START
    INFERENCE_ENGINE_DEPRECATED
    virtual void SetDeviceLoader(const std::string &device, IHeteroDeviceLoader::Ptr loader) noexcept = 0;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @deprecated Use InferenceEngine::Core::QueryNetwork with HETERO device and QueryNetworkResult::supportedLayersMap
     * to set affinities to a network
     * @brief The main goal of this function to set affinity according to the options set for the plugin
     * implementing IHeteroInferencePlugin.
     * This function works only if all affinity in the network are empty.
     * @param network Network object acquired from CNNNetReader
     * @param config Map of configuration settings
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    INFERENCE_ENGINE_DEPRECATED
    virtual StatusCode SetAffinity(
        ICNNNetwork& network,
        const std::map<std::string, std::string> &config,
        ResponseDesc *resp) noexcept = 0;
};

}  // namespace InferenceEngine
