// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp/ie_cnn_network.h"
#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
#include "ie_parameter.hpp"
#include "ie_remote_context.hpp"
#include "so_ptr.hpp"

namespace ov {
class Function;
namespace op {
namespace v0 {
class Parameter;
class Result;
}  // namespace v0
}  // namespace op
}  // namespace ov
namespace InferenceEngine {

class IInferencePlugin;
class IPluginWrapper;
class IInferRequestInternal;
class RemoteContext;
class IVariableStateInternal;
class ICompiledModelWrapper;

/**
 * @interface IExecutableNetworkInternal
 * @brief An internal API of executable network to be implemented by plugin,
 * @ingroup ie_dev_api_exec_network_api
 */
class INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(IExecutableNetworkInternal)
    : public std::enable_shared_from_this<IExecutableNetworkInternal> {
public:
    /**
     * @brief A shared pointer to IExecutableNetworkInternal interface
     */
    using Ptr = std::shared_ptr<IExecutableNetworkInternal>;

    /**
     * @brief      Sets the network inputs info.
     * @param[in]  networkInputs  The network inputs info
     */
    virtual void setNetworkInputs(const InputsDataMap& networkInputs);

    /**
     * @brief      Sets the network outputs data.
     * @param[in]  networkOutputs  The network outputs
     */
    virtual void setNetworkOutputs(const OutputsDataMap& networkOutputs);

    /**
     * @brief      Sets the network parameters
     * @param[in]  params  The network parameters
     */
    virtual void setInputs(const std::vector<std::shared_ptr<const ov::Node>>& params);
    /**
     * @brief      Returns the network parameters
     */
    virtual const std::vector<std::shared_ptr<const ov::Node>>& getInputs() const;
    /**
     * @brief      Sets the network results
     * @param[in]  results  The network results
     */
    virtual void setOutputs(const std::vector<std::shared_ptr<const ov::Node>>& results);
    /**
     * @brief      Returns the network results
     */
    virtual const std::vector<std::shared_ptr<const ov::Node>>& getOutputs() const;

    /**
     * @brief Gets the Executable network output Data node information. The received info is stored in the given Data
     * node.
     * @return out Reference to the ConstOutputsDataMap object
     */
    virtual ConstOutputsDataMap GetOutputsInfo() const;

    /**
     * @brief Gets the Executable network input Data node information. The received info is stored in the given
     * InputsDataMap object.
     * @return inputs Reference to ConstInputsDataMap object.
     */
    virtual ConstInputsDataMap GetInputsInfo() const;

    /**
     * @brief Create an inference request object used to infer the network
     *  Note: the returned request will have allocated input and output blobs (that can be changed later)
     * @return shared_ptr for the created request
     */
    virtual std::shared_ptr<IInferRequestInternal> CreateInferRequest();

    /**
     * @deprecated Use IExecutableNetworkInternal::Export(std::ostream& networkModel)
     * @brief Export the current created executable network so it can be used later in the Import() main API
     * @param modelFileName - path to the location of the exported file
     */
    virtual void Export(const std::string& modelFileName);

    /**
     * @brief Export the current created executable network so it can be used later in the Import() main API
     * @param networkModel - Reference to network model output stream
     */
    virtual void Export(std::ostream& networkModel);

    /**
     * @brief Get executable graph information from a device
     * @return A network object to store executable graph information
     */
    virtual std::shared_ptr<ngraph::Function> GetExecGraphInfo();

    /**
     * @brief      Sets the pointer to plugin internal.
     * @param[in]  plugin  The plugin
     * @note Needed to correctly handle ownership between objects.
     */
    virtual void SetPointerToPlugin(const std::shared_ptr<IInferencePlugin>& plugin);

    /**
     * @brief      Gets the pointer to plugin so.
     * @note Needed to correctly handle ownership between objects.
     * @return A shared pointer to the plugin so
     */
    virtual std::shared_ptr<void> GetPointerToSo();

    /**
     * @brief Sets configuration for current executable network
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    virtual void SetConfig(const std::map<std::string, Parameter>& config);

    /**
     * @brief Gets configuration dedicated to plugin behaviour
     * @param name A config key, can be found in ie_plugin_config.hpp
     * @return A value of config corresponding to config key
     */
    virtual Parameter GetConfig(const std::string& name) const;

    /**
     * @brief Gets general runtime metric for dedicated hardware
     * @param name  A metric name to request
     * @return A metric value corresponding to metric key
     */
    virtual Parameter GetMetric(const std::string& name) const;

    /**
     * @brief Gets the remote context.
     * @return A reference to a context
     */
    virtual std::shared_ptr<RemoteContext> GetContext() const;

    /**
     * @brief Raises the flag that model was loaded from cache
     */
    void loadedFromCache();

    /**
     * @brief Provides an information how model was loaded
     *
     * @return true if model was loaded from cache
     */
    bool isLoadedFromCache() const;

protected:
    virtual ~IExecutableNetworkInternal() = default;

    /**
     * @brief      Creates an inference request internal implementation.
     * @note       The method is called by IExecutableNetworkInternal::CreateInferRequest as
     *             plugin-specific implementation.
     * @param[in]  networkInputs   The network inputs
     * @param[in]  networkOutputs  The network outputs
     * @return     A shared pointer to inference request object.
     */
    virtual std::shared_ptr<IInferRequestInternal> CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                          OutputsDataMap networkOutputs);
    /**
     * @brief      Creates an inference request internal implementation.
     * @note       The method is called by IExecutableNetworkInternal::CreateInferRequest as
     *             plugin-specific implementation.
     * @param[in]  inputs   The function inputs
     * @param[in]  outputs  The function outputs
     * @return     A shared pointer to inference request object.
     */
    virtual std::shared_ptr<IInferRequestInternal> CreateInferRequestImpl(
        const std::vector<std::shared_ptr<const ov::Node>>& inputs,
        const std::vector<std::shared_ptr<const ov::Node>>& outputs);

    InferenceEngine::InputsDataMap _networkInputs;    //!< Holds information about network inputs info
    InferenceEngine::OutputsDataMap _networkOutputs;  //!< Holds information about network outputs data
    std::vector<std::shared_ptr<const ov::Node>> _parameters;
    std::vector<std::shared_ptr<const ov::Node>> _results;

    /**
     * @brief A pointer to a IInferencePlugin interface.
     * @note Needed to correctly handle ownership between objects.
     */
    std::shared_ptr<IInferencePlugin> _plugin;

    /**
     * @brief A pointer to a plugin library.
     * @note Needed to correctly handle ownership between objects.
     */
    std::shared_ptr<void> _so;

    /**
     * @brief If true, it means that model was loaded from cache
     */
    bool _loadedFromCache = false;

    friend InferenceEngine::ICompiledModelWrapper;
    friend InferenceEngine::IPluginWrapper;
};

/**
 * @brief SoPtr to IExecutableNetworkInternal.
 */
using SoExecutableNetworkInternal = ov::SoPtr<IExecutableNetworkInternal>;

}  // namespace InferenceEngine
