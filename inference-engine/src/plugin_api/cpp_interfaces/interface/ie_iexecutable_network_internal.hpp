// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>
#include <ie_iinfer_request.hpp>
#include <ie_parameter.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {

/**
 * @interface IExecutableNetworkInternal
 * @brief An internal API of executable network to be implemented by plugin,
 * which is used in ExecutableNetworkBase forwarding mechanism.
 * @ingroup ie_dev_api_exec_network_api
 */
class IExecutableNetworkInternal {
public:
    /**
     * @brief A shared pointer to IExecutableNetworkInternal interface
     */
    typedef std::shared_ptr<IExecutableNetworkInternal> Ptr;

    /**
     * @brief      Destroys the object.
     */
    virtual ~IExecutableNetworkInternal() = default;

    /**
     * @brief Gets the Executable network output Data node information. The received info is stored in the given Data
     * node. This method need to be called to find output names for using them later during filling of a map of blobs
     * passed later to InferenceEngine::IInferencePlugin::Infer()
     * @return out Reference to the ConstOutputsDataMap object
     */
    virtual ConstOutputsDataMap GetOutputsInfo() const = 0;

    /**
     * @brief Gets the Executable network input Data node information. The received info is stored in the given
     * InputsDataMap object. This method need to be called to find out input names for using them later during filling
     * of a map of blobs passed later to InferenceEngine::IInferencePlugin::Infer()
     * @return inputs Reference to ConstInputsDataMap object.
     */
    virtual ConstInputsDataMap GetInputsInfo() const = 0;

    /**
     * @brief Create an inference request object used to infer the network
     *  Note: the returned request will have allocated input and output blobs (that can be changed later)
     * @return shared_ptr for the created request
     */
    virtual IInferRequest::Ptr CreateInferRequest() = 0;

    /**
     * @deprecated Use IExecutableNetworkInternal::Export(std::ostream& networkModel)
     * @brief Export the current created executable network so it can be used later in the Import() main API
     * @param modelFileName - path to the location of the exported file
     */
    virtual void Export(const std::string& modelFileName) = 0;

    /**
     * @brief Export the current created executable network so it can be used later in the Import() main API
     * @param networkModel - Reference to network model output stream
     */
    virtual void Export(std::ostream& networkModel) = 0;

    /**
     * @brief Get executable graph information from a device
     * @return A network object to store executable graph information
     */
    virtual CNNNetwork GetExecGraphInfo() = 0;

    /**
     * @deprecated Need to implement GetVariablesInfo for ExecutableNetwork
     * @brief Queries memory states.
     * @return Returns memory states
     */
    virtual std::vector<IVariableStateInternal::Ptr> QueryState() = 0;

    /**
     * @brief Sets configuration for current executable network
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    virtual void SetConfig(const std::map<std::string, Parameter>& config) = 0;

    /**
     * @brief Gets configuration dedicated to plugin behaviour
     * @param name A config key, can be found in ie_plugin_config.hpp
     * @return A value of config corresponding to config key
     */
    virtual Parameter GetConfig(const std::string& name) const = 0;

    /**
     * @brief Gets general runtime metric for dedicated hardware
     * @param name  A metric name to request
     * @return A metric value corresponding to metric key
     */
    virtual Parameter GetMetric(const std::string& name) const = 0;

    /**
     * @brief Gets the remote context.
     * @return A reference to a context
     */
    virtual RemoteContext::Ptr GetContext() const = 0;
};

}  // namespace InferenceEngine
