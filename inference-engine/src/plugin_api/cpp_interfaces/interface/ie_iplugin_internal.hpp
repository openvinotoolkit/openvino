// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Inference Engine plugin API wrapper, to be used by particular implementors
 * @file ie_iplugin_internal.hpp
 */

#pragma once

#include <ie_iextension.h>
#include <ie_input_info.hpp>
#include <ie_icnn_network.hpp>
#include <ie_icore.hpp>
#include <ie_parameter.hpp>
#include <ie_iexecutable_network.hpp>
#include <ie_remote_context.hpp>

#include <blob_factory.hpp>

#include <istream>
#include <map>
#include <memory>
#include <string>

namespace InferenceEngine {

/**
 * @brief      Copies preprocess info
 *
 * @param[in]  from  PreProcessInfo to copy from
 * @param      to    PreProcessInfo to copy to
 */
static void copyPreProcess(const PreProcessInfo& from, PreProcessInfo& to) {
    to = from;
    if (from.getMeanVariant() == MEAN_IMAGE) {
        for (size_t i = 0; i < from.getNumberOfChannels(); i++) {
            auto& from_blob = from[i]->meanData;
            auto to_blob = make_blob_with_precision(from[i]->meanData->getTensorDesc());
            to_blob->allocate();
            ie_memcpy(to_blob->buffer(), to_blob->byteSize(), from_blob->cbuffer(), from_blob->byteSize());

            to.setMeanImageForChannel(to_blob, i);
        }
    }
}

/**
 * @brief      Copies InputInfo and output Data
 *
 * @param[in]  networkInputs    The network inputs to copy from
 * @param[in]  networkOutputs   The network outputs to copy from
 * @param      _networkInputs   The network inputs to copy to
 * @param      _networkOutputs  The network outputs to copy to
 */
static void copyInputOutputInfo(const InputsDataMap & networkInputs, const OutputsDataMap & networkOutputs,
                                InputsDataMap & _networkInputs, OutputsDataMap & _networkOutputs) {
    _networkInputs.clear();
    _networkOutputs.clear();

    for (const auto& it : networkInputs) {
        InputInfo::Ptr newPtr;
        if (it.second) {
            newPtr.reset(new InputInfo());
            copyPreProcess(it.second->getPreProcess(), newPtr->getPreProcess());
            DataPtr newData(new Data(*it.second->getInputData()));
            newPtr->setInputData(newData);
        }
        _networkInputs[it.first] = newPtr;
    }
    for (const auto& it : networkOutputs) {
        DataPtr newData;
        if (it.second) {
            newData.reset(new Data(*it.second));
        }
        _networkOutputs[it.first] = newData;
    }
}

/**
 * @interface IInferencePlugin
 * @brief An API of plugin to be implemented by a plugin
 * @ingroup ie_dev_api_plugin_api
 */
class IInferencePlugin : public details::IRelease,
                         public std::enable_shared_from_this<IInferencePlugin> {
    class VersionStore : public Version {
        std::string _dsc;
        std::string _buildNumber;

        void copyFrom(const Version & v) {
            _dsc = v.description;
            _buildNumber = v.buildNumber;
            description = _dsc.c_str();
            buildNumber = _buildNumber.c_str();
            apiVersion = v.apiVersion;
        }

    public:
        VersionStore() = default;

        explicit VersionStore(const Version& v) {
            copyFrom(v);
        }

        VersionStore & operator = (const VersionStore & v) {
            if (&v != this) {
                copyFrom(v);
            }
            return *this;
        }
    } _version;

protected:
    /**
     * @brief      Destroys the object.
     */
    ~IInferencePlugin() override = default;

public:
    /**
     * @brief A shared pointer to IInferencePlugin interface
     */
    using Ptr = std::shared_ptr<IInferencePlugin>;

    /**
     * @brief Sets a plugin version
     * @param version A version to set
     */
    void SetVersion(const Version & version) {
        _version = VersionStore(version);
    }

    /**
     * @brief Gets a plugin version
     * @return A const InferenceEngine::Version object
     */
    Version GetVersion() const {
        return _version;
    }

    void Release() noexcept override {
        delete this;
    }

    /**
     * @brief      Provides a name of a plugin
     * @return     The name.
     */
    virtual std::string GetName() const noexcept = 0;

    /**
     * @brief      Sets a name for a plugin 
     * @param[in]  name  The name
     */
    virtual void SetName(const std::string& name) noexcept = 0;

    /**
     * @brief Creates an executable network from an pares network object, users can create as many networks as they need
     * and use them simultaneously (up to the limitation of the HW resources)
     * @param network A network object acquired from InferenceEngine::Core::ReadNetwork
     * @param config A string-string map of config parameters relevant only for this load operation
     * @return Created Executable Network object
     */
    virtual ExecutableNetwork LoadNetwork(const ICNNNetwork& network,
                                          const std::map<std::string, std::string>& config) = 0;

    /**
     * @brief Creates an executable network from network object, on specified remote context
     * @param network - a network object acquired from InferenceEngine::Core::ReadNetwork
     * @param config string-string map of config parameters relevant only for this load operation
     * @param context - a pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @return Created Executable Network object
     */
    virtual ExecutableNetwork LoadNetwork(const ICNNNetwork& network,
                                          const std::map<std::string, std::string>& config,
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
     * @brief Creates an executable network from an previously exported network
     * @param modelFileName - path to the location of the exported file
     * @param config A string -> string map of parameters
     * @return A reference to a shared ptr of the returned network interface
     */
    virtual IExecutableNetwork::Ptr ImportNetwork(const std::string& modelFileName,
                                                  const std::map<std::string, std::string>& config) = 0;

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
     * @brief Sets pointer to ICore interface
     * @param core Pointer to Core interface
     */
    virtual void SetCore(ICore* core) noexcept = 0;

    /**
     * @brief Gets reference to ICore interface
     * @return Reference to ICore interface
     */
    virtual ICore* GetCore() const noexcept = 0;

    /**
     * @brief      Queries a plugin about supported layers in network
     * @param[in]  network  The network object to query
     * @param[in]  config   The map of configuration parameters
     * @param      res      The result of query operator containing supported layers map
     */
    virtual void QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                              QueryNetworkResult& res) const = 0;
};

}  // namespace InferenceEngine

/**
 * @def IE_DEFINE_PLUGIN_CREATE_FUNCTION(PluginType, version)
 * @brief Defines the exported `CreatePluginEngine` function which is used to create a plugin instance
 * @ingroup ie_dev_api_plugin_api
 */
#define IE_DEFINE_PLUGIN_CREATE_FUNCTION(PluginType, version, ...)                       \
    INFERENCE_PLUGIN_API(InferenceEngine::StatusCode) CreatePluginEngine(                \
            InferenceEngine::IInferencePlugin *&plugin,                                  \
            InferenceEngine::ResponseDesc *resp) noexcept {                              \
        try {                                                                            \
            plugin = new PluginType(__VA_ARGS__);                                        \
            plugin->SetVersion(version);                                                 \
            return OK;                                                                   \
        }                                                                                \
        catch (std::exception &ex) {                                                     \
            return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what(); \
        }                                                                                \
    }
