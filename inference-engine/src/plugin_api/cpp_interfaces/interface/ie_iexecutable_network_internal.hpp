// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ie_parameter.hpp>
#include <ie_remote_context.hpp>
#include <cpp/ie_cnn_network.h>
#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>
#include <details/ie_so_pointer.hpp>

namespace InferenceEngine {

class IInferencePlugin;
class IInferRequestInternal;
class RemoteContext;
class IVariableStateInternal;

/**
 * @interface IExecutableNetworkInternal
 * @brief An internal API of executable network to be implemented by plugin,
 * @ingroup ie_dev_api_exec_network_api
 */
class INFERENCE_ENGINE_API_CLASS(IExecutableNetworkInternal) : public std::enable_shared_from_this<IExecutableNetworkInternal> {
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
    virtual CNNNetwork GetExecGraphInfo();

    /**
     * @deprecated Need to implement GetVariablesInfo for ExecutableNetwork
     * @brief Queries memory states.
     * @return Returns memory states
     */
    virtual std::vector<std::shared_ptr<IVariableStateInternal>> QueryState();

    /**
     * @brief      Sets the pointer to plugin internal.
     * @param[in]  plugin  The plugin
     * @note Needed to correctly handle ownership between objects.
     */
    virtual void SetPointerToPlugin(const std::shared_ptr<IInferencePlugin>& plugin);

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

protected:
    ~IExecutableNetworkInternal() = default;

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

    InferenceEngine::InputsDataMap _networkInputs;  //!< Holds information about network inputs info
    InferenceEngine::OutputsDataMap _networkOutputs;  //!< Holds information about network outputs data

    /**
     * @brief A pointer to a IInferencePlugin interface.
     * @note Needed to correctly handle ownership between objects.
     */
    std::shared_ptr<IInferencePlugin> _plugin;
};

/**
 * @brief SOPointer to IExecutableNetworkInternal.
 */
using SoExecutableNetworkInternal = details::SOPointer<IExecutableNetworkInternal>;

}  // namespace InferenceEngine
