// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief inference engine plugin API wrapper, to be used by particular implementors
 * \file ie_plugin_base.hpp
 */

#pragma once

#include <ie_iextension.h>

#include <cpp/ie_executable_network.hpp>
#include <ie_icnn_network.hpp>
#include <ie_icore.hpp>
#include <ie_iexecutable_network.hpp>
#include <ie_remote_context.hpp>
#include <istream>
#include <limits>
#include <map>
#include <memory>
#include <string>

namespace InferenceEngine {

/**
 * @brief minimum API to be implemented by plugin, which is used in PluginBase forwarding mechanism
 */
class IInferencePluginInternal {
public:
    using Ptr = std::shared_ptr<IInferencePluginInternal>;

    virtual ~IInferencePluginInternal() = default;

    virtual std::string GetName() const noexcept = 0;
    virtual void SetName(const std::string& name) noexcept = 0;

    /**
     * @brief Creates an executable network from an pares network object, users can create as many networks as they need
     * and use them simultaneously (up to the limitation of the HW resources)
     * @param executableNetwork - a reference to a shared ptr of the returned network interface
     * @param network - a network object acquired from CNNNetReader
     * @param config string-string map of config parameters relevant only for this load operation
     */
    virtual void LoadNetwork(IExecutableNetwork::Ptr& executableNetwork, ICNNNetwork& network,
                             const std::map<std::string, std::string>& config) = 0;

    /**
     * @brief Creates an executable network from network object, on specified remote context
     * @param network - a network object acquired from CNNNetReader
     * @param config string-string map of config parameters relevant only for this load operation
     * @param context - a pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @return Created Executable Network object
     */
    virtual ExecutableNetwork LoadNetwork(ICNNNetwork& network, const std::map<std::string, std::string>& config,
                                          RemoteContext::Ptr context) = 0;
    /**
     * @brief Registers extension within plugin
     * @param extension - pointer to already loaded extension
     */
    virtual void AddExtension(InferenceEngine::IExtensionPtr extension) = 0;

    /**
     * @brief Sets configuration for plugin, acceptable keys can be found in ie_plugin_config.hpp
     * @param config string-string map of config parameters
     */
    virtual void SetConfig(const std::map<std::string, std::string>& config) = 0;

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

    virtual RemoteContext::Ptr CreateContext(const ParamMap& params) = 0;
    virtual RemoteContext::Ptr GetDefaultContext() = 0;

    /**
     * @brief Creates an executable network from an previously exported network
     * @param ret - a reference to a shared ptr of the returned network interface
     * @param modelFileName - path to the location of the exported file
     */
    virtual IExecutableNetwork::Ptr ImportNetwork(const std::string& modelFileName,
                                                  const std::map<std::string, std::string>& config) = 0;

    /**
     * @brief Creates an executable network from an previously exported network using plugin implementation
     *        and removes Inference Engine magic and plugin name
     * @param networkModel - Reference to network model output stream
     * @return An Executable network
     */
    virtual ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                            const std::map<std::string, std::string>& config) {
        ExportMagic magic = {};
        auto currentPos = networkModel.tellg();
        networkModel.read(magic.data(), magic.size());
        auto exportedWithName = (exportMagic == magic);
        if (exportedWithName) {
            networkModel.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        } else {
            networkModel.seekg(currentPos, networkModel.beg);
        }
        return ImportNetworkImpl(networkModel, config);
    }

    /**
     * @brief Creates an executable network from an previously exported network
     * @param networkModel - Reference to network model output stream
     * @return An Executable network
     */
    virtual ExecutableNetwork ImportNetworkImpl(std::istream& networkModel,
                                                const std::map<std::string, std::string>& config) = 0;

    /**
     * @brief Sets logging callback
     * Logging is used to track what is going on inside
     * @param listener - logging sink
     */
    virtual void SetLogCallback(IErrorListener& listener) = 0;

    /**
     * @brief Sets pointer to ICore interface
     * @param core Pointer to Core interface
     */
    virtual void SetCore(ICore* core) noexcept = 0;

    /**
     * @brief Gets refernce to ICore interface
     * @return Reference to core interface
     */
    virtual const ICore* GetCore() const noexcept = 0;

    virtual void QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                              QueryNetworkResult& res) const = 0;
};

}  // namespace InferenceEngine
