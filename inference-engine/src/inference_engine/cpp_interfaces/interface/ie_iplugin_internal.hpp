// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief inference engine plugin API wrapper, to be used by particular implementors
 * \file ie_plugin_base.hpp
 */

#pragma once

#include <memory>
#include <map>
#include <string>
#include <ie_icnn_network.hpp>
#include <ie_iexecutable_network.hpp>
#include <ie_iextension.h>

namespace InferenceEngine {

/**
 * @brief minimum API to be implemented by plugin, which is used in PluginBase forwarding mechanism
 */
class IInferencePluginInternal {
public:
    virtual ~IInferencePluginInternal() = default;

    /**
     * @deprecated use LoadNetwork with 4 parameters (executable network, cnn network, config, response)
     * @brief Loads a pre-built network with weights to the engine so it will be ready for inference
     * @param network - a network object acquired from CNNNetReader
     */
    virtual void LoadNetwork(ICNNNetwork &network) = 0;

    /**
     * @brief Creates an executable network from an pares network object, users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the HW resources)
     * @param executableNetwork - a reference to a shared ptr of the returned network interface
     * @param network - a network object acquired from CNNNetReader
     * @param config string-string map of config parameters relevant only for this load operation
     */
    virtual void LoadNetwork(IExecutableNetwork::Ptr &executableNetwork,
                             ICNNNetwork &network,
                             const std::map<std::string, std::string> &config) = 0;

    /**
     * @deprecated use Infer() working with multiple inputs and outputs
     * @brief Infers an image(s)
     * Input and output dimension depend on topology.
     *     As an example for classification topologies use a 4D Blob as input (batch, channels, width,
     *             height) and get a 1D blob as output (scoring probability vector). To Infer a batch,
     *             use a 4D Blob as input and get a 2D blob as output in both cases the method will
     *             allocate the resulted blob
     * @param input - any TBlob<> object that contains the data to infer. the type of TBlob must correspond to the network input precision and size.
     * @param result - a related TBlob<> object that will contain the result of the inference action, typically this should be a float blob.
               The blob does not need to be allocated or initialized, the engine will allocate the relevant data
     */
    virtual void Infer(const Blob &input, Blob &result) = 0;

    /**
     * @deprecated load IExecutableNetwork to create IInferRequest.
     * @brief Infer data: to Infer tensors. Input and ouput dimension depend on topology.
     *     As an example for classification topologies use a 4D Blob as input (batch, chanels, width,
     *             height) and get a 1D blob as output (scoring probability vector). To Infer a batch,
     *             use a 4D Blob as input and get a 2D blob as output in both cases the method will
     *             allocate the resulted blob
     * @param input - map of input blobs accessed by input names
     * @param result - map of output blobs accessed by output names
     */
    virtual void Infer(const BlobMap &input, BlobMap &result) = 0;

    /**
     * @deprecated use IInferRequest to get performance measures
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer.
     *  Note: not all plugins may provide meaningful data
     *  @param perfMap - a map of layer names to profiling information for that layer.
     */
    virtual void GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) = 0;

    /**
     * @brief Registers extension within plugin
     * @param extension - pointer to already loaded extension
     */
    virtual void AddExtension(InferenceEngine::IExtensionPtr extension) = 0;

    /**
     * @brief Sets configuration for plugin, acceptable keys can be found in ie_plugin_config.hpp
     * @param config string-string map of config parameters
     */
    virtual void SetConfig(const std::map<std::string, std::string> &config) = 0;

    /**
     * @brief Creates an executable network from an previously exported network
     * @param ret - a reference to a shared ptr of the returned network interface
     * @param modelFileName - path to the location of the exported file
     */
    virtual IExecutableNetwork::Ptr ImportNetwork(const std::string &modelFileName, const std::map<std::string, std::string> &config) = 0;

    /**
     * @brief Sets logging callback
     * Logging is used to track what is going on inside
     * @param listener - logging sink
     */
    virtual void SetLogCallback(IErrorListener &listener) = 0;

    /**
     * @depricated Use the version with config parameter
     */
    virtual void QueryNetwork(const ICNNNetwork& network, QueryNetworkResult& res) const = 0;

    virtual void QueryNetwork(const ICNNNetwork &network, const std::map<std::string, std::string>& config, QueryNetworkResult &res) const = 0;
};

}  // namespace InferenceEngine
