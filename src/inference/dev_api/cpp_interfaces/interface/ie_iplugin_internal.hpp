// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Inference Engine plugin API wrapper, to be used by particular implementors
 * @file ie_iplugin_internal.hpp
 */

#pragma once

#include <istream>
#include <map>
#include <memory>
#include <string>

#include "blob_factory.hpp"
#include "cpp/ie_cnn_network.h"
#include "ie_iextension.h"
#include "ie_input_info.hpp"
#include "ie_parameter.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/util/pp.hpp"
#include "so_ptr.hpp"

namespace InferenceEngine {

class ExecutorManager;
class IExecutableNetworkInternal;
class RemoteContext;
class IExtension;
class ICore;

/**
 * @brief      Copies preprocess info
 *
 * @param[in]  from  PreProcessInfo to copy from
 * @return     copy of preprocess info
 */
INFERENCE_ENGINE_API_CPP(PreProcessInfo) copyPreProcess(const PreProcessInfo& from);

/**
 * @brief       Copies the values of `std::string` indexed map and apply const cast
 *
 * @param[in]   map map to copy
 * @return      map that contains pointers to constant values
 */
template <typename T>
std::map<std::string, std::shared_ptr<const T>> constMapCast(const std::map<std::string, std::shared_ptr<T>>& map) {
    std::map<std::string, std::shared_ptr<const T>> res;
    for (auto&& v : map)
        res.emplace(v.first, std::const_pointer_cast<const T>(v.second));
    return res;
}

/**
 * @brief       Copies the values of `std::string` indexed map and apply const cast
 *
 * @param[in]   map map to copy
 * @return      map that contains pointers to values
 */
template <typename T>
std::map<std::string, std::shared_ptr<T>> constMapCast(const std::map<std::string, std::shared_ptr<const T>>& map) {
    std::map<std::string, std::shared_ptr<T>> res;
    for (auto&& v : map)
        res.emplace(v.first, std::const_pointer_cast<T>(v.second));
    return res;
}

/**
 * @brief      Copies InputInfo
 *
 * @param[in]  networkInputs    The network inputs to copy from
 * @return copy of network inputs
 */
INFERENCE_ENGINE_API_CPP(InputsDataMap) copyInfo(const InputsDataMap& networkInputs);

/**
 * @brief      Copies OutputsData
 *
 * @param[in]  networkInputs    network outputs to copy from
 * @return copy of network outputs
 */
INFERENCE_ENGINE_API_CPP(OutputsDataMap) copyInfo(const OutputsDataMap& networkOutputs);

/**
 * @brief Set input and output information to executable network. This method is used to
 * set additional information to InferenceEngine::IExecutableNetworkInternal created by device plugin.
 * @param exeNetwork Executable network object
 * @param function Model with initial execution info
 */
INFERENCE_ENGINE_API_CPP(void)
SetExeNetworkInfo(const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork,
                  const std::shared_ptr<const ov::Model>& function,
                  bool new_api);

/**
 * @brief Returns set of nodes which were removed after transformation.
 * If originalFunction contains node1 and transformedFunction does not
 * contains node1 in ops list, node1 will be returned.
 * @param originalFunction Original network
 * @param transformedFunction Transformed network
 * @return Set of strings which contains removed node names
 */
INFERENCE_ENGINE_API_CPP(std::unordered_set<std::string>)
GetRemovedNodes(const std::shared_ptr<const ov::Model>& originalFunction,
                const std::shared_ptr<const ov::Model>& transformedFunction);

/**
 * @brief Returns set of nodes from original model which are
 * determined as supported after applied transformation pipeline.
 * @param model Original model
 * @param transform Transformation pipeline function
 * @param is_node_supported Function returning whether node is supported or not
 * @return Set of strings which contains supported node names
 */
INFERENCE_ENGINE_API_CPP(std::unordered_set<std::string>)
GetSupportedNodes(const std::shared_ptr<const ov::Model>& model,
                  std::function<void(std::shared_ptr<ov::Model>&)> transform,
                  std::function<bool(const std::shared_ptr<ngraph::Node>)> is_node_supported);

/**
 * @interface IInferencePlugin
 * @brief An API of plugin to be implemented by a plugin
 * @ingroup ie_dev_api_plugin_api
 */
class INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(IInferencePlugin)
    : public std::enable_shared_from_this<IInferencePlugin> {
    class VersionStore : public Version {
        void copyFrom(const Version& v);

    public:
        VersionStore() = default;

        explicit VersionStore(const Version& v);

        VersionStore& operator=(const VersionStore& v);
    } _version;

public:
    /**
     * @brief A shared pointer to IInferencePlugin interface
     */
    using Ptr = std::shared_ptr<IInferencePlugin>;

    /**
     * @brief Sets a plugin version
     * @param version A version to set
     */
    void SetVersion(const Version& version);

    /**
     * @brief Gets a plugin version
     * @return A const InferenceEngine::Version object
     */
    const Version& GetVersion() const;

    /**
     * @brief      Provides a name of a plugin
     * @return     The name.
     */
    virtual std::string GetName() const noexcept;

    /**
     * @brief      Sets a name for a plugin
     * @param[in]  name  The name
     */
    virtual void SetName(const std::string& name) noexcept;

    /**
     * @brief Creates an executable network from an pares network object, users can create as many networks as they need
     * and use them simultaneously (up to the limitation of the HW resources)
     * @param network A network object acquired from InferenceEngine::Core::ReadNetwork
     * @param config A string-string map of config parameters relevant only for this load operation
     * @return Created Executable Network object
     */
    virtual std::shared_ptr<IExecutableNetworkInternal> LoadNetwork(const CNNNetwork& network,
                                                                    const std::map<std::string, std::string>& config);

    /**
     * @brief Creates an executable network from network object, on specified remote context
     * @param network A network object acquired from InferenceEngine::Core::ReadNetwork
     * @param config string-string map of config parameters relevant only for this load operation
     * @param context A pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @return Created Executable Network object
     */
    virtual std::shared_ptr<IExecutableNetworkInternal> LoadNetwork(const CNNNetwork& network,
                                                                    const std::map<std::string, std::string>& config,
                                                                    const std::shared_ptr<RemoteContext>& context);

    /**
     * @brief Creates an executable network from model file path
     * @param modelPath A path to model
     * @param config A string-string map of config parameters relevant only for this load operation
     * @return Created Executable Network object
     */
    virtual ov::SoPtr<IExecutableNetworkInternal> LoadNetwork(const std::string& modelPath,
                                                              const std::map<std::string, std::string>& config);

    /**
     * @brief Registers extension within plugin
     * @param extension - pointer to already loaded extension
     */
    virtual void AddExtension(const std::shared_ptr<IExtension>& extension);

    /**
     * @brief Sets configuration for plugin, acceptable keys can be found in ie_plugin_config.hpp
     * @param config string-string map of config parameters
     */
    virtual void SetConfig(const std::map<std::string, std::string>& config);

    /**
     * @brief Sets configuration for plugin, acceptable keys can be found in openvino/runtime/properties.hpp
     * @param config  ov::AnyMap of config parameters
     */
    virtual void SetProperties(const ov::AnyMap& config);

    /**
     * @brief Gets configuration dedicated to plugin behaviour
     * @param name  - value of config corresponding to config key
     * @param options - configuration details for config
     * @return Value of config corresponding to config key
     */
    virtual Parameter GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const;

    /**
     * @brief Gets general runtime metric for dedicated hardware
     * @param name  - metric name to request
     * @param options - configuration details for metric
     * @return Metric value corresponding to metric key
     */
    virtual Parameter GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const;

    /**
     * @brief      Creates a remote context instance based on a map of parameters
     * @param[in]  params  The map of parameters
     * @return     A remote context object
     */
    virtual std::shared_ptr<RemoteContext> CreateContext(const ParamMap& params);

    /**
     * @brief      Provides a default remote context instance if supported by a plugin
     * @param[in]  params  The map of parameters
     * @return     The default context.
     */
    virtual std::shared_ptr<RemoteContext> GetDefaultContext(const ParamMap& params);

    /**
     * @deprecated Use ImportNetwork(std::istream& networkModel, const std::map<std::string, std::string>& config)
     * @brief Creates an executable network from an previously exported network
     * @param modelFileName - path to the location of the exported file
     * @param config A string -> string map of parameters
     * @return An Executable network
     */
    virtual std::shared_ptr<IExecutableNetworkInternal> ImportNetwork(const std::string& modelFileName,
                                                                      const std::map<std::string, std::string>& config);

    /**
     * @brief Creates an executable network from an previously exported network using plugin implementation
     *        and removes Inference Engine magic and plugin name
     * @param networkModel Reference to network model output stream
     * @param config A string -> string map of parameters
     * @return An Executable network
     */
    virtual std::shared_ptr<IExecutableNetworkInternal> ImportNetwork(std::istream& networkModel,
                                                                      const std::map<std::string, std::string>& config);

    /**
     * @brief Creates an executable network from an previously exported network using plugin implementation
     *        and removes Inference Engine magic and plugin name
     * @param networkModel Reference to network model output stream
     * @param context A pointer to plugin context derived from RemoteContext class used to
     *        execute the network
     * @param config A string -> string map of parameters
     * @return An Executable network
     */
    virtual std::shared_ptr<IExecutableNetworkInternal> ImportNetwork(std::istream& networkModel,
                                                                      const std::shared_ptr<RemoteContext>& context,
                                                                      const std::map<std::string, std::string>& config);

    /**
     * @brief Sets pointer to ICore interface
     * @param core Pointer to Core interface
     */
    virtual void SetCore(std::weak_ptr<InferenceEngine::ICore> core);

    /**
     * @brief Gets reference to ICore interface
     * @return Reference to ICore interface
     */
    virtual std::shared_ptr<InferenceEngine::ICore> GetCore() const noexcept;

    /**
     * @brief Provides an information about used API
     * @return true if new API is used
     */
    bool IsNewAPI() const noexcept;

    /**
     * @brief Gets reference to tasks execution manager
     * @return Reference to ExecutorManager interface
     */
    const std::shared_ptr<ExecutorManager>& executorManager() const;

    /**
     * @brief      Queries a plugin about supported layers in network
     * @param[in]  network  The network object to query
     * @param[in]  config   The map of configuration parameters
     * @return     The result of query operator containing supported layers map
     */
    virtual QueryNetworkResult QueryNetwork(const CNNNetwork& network,
                                            const std::map<std::string, std::string>& config) const;

protected:
    IInferencePlugin();
    virtual ~IInferencePlugin() = default;

    /**
     * @brief Creates an executable network from a parsed network object, users can create as many networks as they need
     *        and use them simultaneously (up to the limitation of the HW resources)
     * @note The function is used in
     * InferencePluginInternal::LoadNetwork(const CNNNetwork&, const std::map<std::string, std::string>&)
     * which performs common steps first and calls this plugin-dependent method implementation after.
     * @param network A network object
     * @param config string-string map of config parameters relevant only for this load operation
     * @return Shared pointer to the ExecutableNetwork object
     */
    virtual std::shared_ptr<IExecutableNetworkInternal> LoadExeNetworkImpl(
        const CNNNetwork& network,
        const std::map<std::string, std::string>& config);

    /**
     * @brief Creates an executable network using remote context from a parsed network object,
     * users can create as many networks as they need and use them simultaneously (up to the limitation of the HW
     * resources)
     * @note The function is used in
     * InferencePluginInternal::LoadNetwork(const CNNNetwork&, const std::map<std::string, std::string>&,
     * RemoteContext::Ptr) which performs common steps first and calls this plugin-dependent method implementation
     * after.
     * @param network A network object
     * @param context A remote context
     * @param config string-string map of config parameters relevant only for this load operation
     * @return Shared pointer to the ExecutableNetwork object
     */
    virtual std::shared_ptr<IExecutableNetworkInternal> LoadExeNetworkImpl(
        const CNNNetwork& network,
        const std::shared_ptr<RemoteContext>& context,
        const std::map<std::string, std::string>& config);

    /**
     * @brief Set input and output information to executable network. This method is used to
     * set addtional information to InferenceEngine::IExecutableNetworkInternal create by device plugin.
     * @param exeNetwork An executable network object to set information to
     * @param inputs An input information to set
     * @param outputs An output information to set
     * @param function Function with initial execution info
     */
    void SetExeNetworkInfo(const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork,
                           const ConstInputsDataMap& inputs,
                           const ConstOutputsDataMap& outputs);

    /**
     * @brief Set input and output information to executable network. This method is used to
     * set additional information to InferenceEngine::IExecutableNetworkInternal create by device plugin.
     * @param function Function with initial execution info
     */
    void SetExeNetworkInfo(const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork,
                           const std::shared_ptr<const ov::Model>& function);

    std::string _pluginName;                            //!< A device name that plugins enables
    std::map<std::string, std::string> _config;         //!< A map config keys -> values
    std::weak_ptr<InferenceEngine::ICore> _core;        //!< A pointer to ICore interface
    std::shared_ptr<ExecutorManager> _executorManager;  //!< A tasks execution manager
    bool _isNewAPI;                                     //!< A flag which shows used API
};

/**
 * @private
 */
using CreatePluginEngineFunc = void(std::shared_ptr<::ov::IPlugin>&);

/**
 * @private
 */
using CreateExtensionFunc = void(std::shared_ptr<IExtension>&);

/**
 * @def IE_CREATE_PLUGIN
 * @brief Defines a name of a function creating plugin instance
 * @ingroup ie_dev_api_plugin_api
 */
#ifndef IE_CREATE_PLUGIN
#    define IE_CREATE_PLUGIN CreatePluginEngine
#endif

/**
 * @private
 */
constexpr static const auto create_plugin_function = OV_PP_TOSTRING(IE_CREATE_PLUGIN);

INFERENCE_ENGINE_API_CPP(std::shared_ptr<::ov::IPlugin>)
convert_plugin(const std::shared_ptr<InferenceEngine::IInferencePlugin>& from);

}  // namespace InferenceEngine

/**
 * @def IE_DEFINE_PLUGIN_CREATE_FUNCTION(PluginType, version)
 * @brief Defines the exported `IE_CREATE_PLUGIN` function which is used to create a plugin instance
 * @ingroup ie_dev_api_plugin_api
 */
#define IE_DEFINE_PLUGIN_CREATE_FUNCTION(PluginType, version, ...)                    \
    INFERENCE_PLUGIN_API(void)                                                        \
    IE_CREATE_PLUGIN(::std::shared_ptr<::ov::IPlugin>& plugin) noexcept(false);       \
    void IE_CREATE_PLUGIN(::std::shared_ptr<::ov::IPlugin>& plugin) noexcept(false) { \
        std::shared_ptr<::InferenceEngine::IInferencePlugin> ie_plugin;               \
        try {                                                                         \
            ie_plugin = ::std::make_shared<PluginType>(__VA_ARGS__);                  \
        } catch (const InferenceEngine::Exception&) {                                 \
            throw;                                                                    \
        } catch (const std::exception& ex) {                                          \
            IE_THROW() << ex.what();                                                  \
        } catch (...) {                                                               \
            IE_THROW(Unexpected);                                                     \
        }                                                                             \
        ie_plugin->SetVersion(version);                                               \
        plugin = convert_plugin(ie_plugin);                                           \
    }

/**
 * @private
 */
#define IE_DEFINE_PLUGIN_CREATE_FUNCTION_DECLARATION(_IE_CREATE_PLUGIN_FUNC) \
    INFERENCE_PLUGIN_API(void)                                               \
    _IE_CREATE_PLUGIN_FUNC(::std::shared_ptr<::ov::IPlugin>& plugin) noexcept(false)

/**
 * @private
 */
#define IE_DEFINE_EXTENSION_CREATE_FUNCTION_DECLARATION(_IE_CREATE_EXTENSION_FUNC) \
    INFERENCE_EXTENSION_API(void)                                                  \
    _IE_CREATE_EXTENSION_FUNC(::InferenceEngine::IExtensionPtr& ext)
