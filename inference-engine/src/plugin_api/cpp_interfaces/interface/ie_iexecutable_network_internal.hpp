// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_imemory_state_internal.hpp>
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
     * @param req - shared_ptr for the created request
     */
    virtual void CreateInferRequest(IInferRequest::Ptr& req) = 0;

    /**
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
     * @param graphPtr network ptr to store executable graph information
     */
    virtual void GetExecGraphInfo(ICNNNetwork::Ptr& graphPtr) = 0;

    /**
     * @brief Queries memory states.
     * @return Returns memory states
     */
    virtual std::vector<IMemoryStateInternal::Ptr> QueryState() = 0;

    /**
     * @brief Sets configuration for current executable network
     * @param config Map of pairs: (config parameter name, config parameter value)
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     */
    virtual void SetConfig(const std::map<std::string, Parameter>& config, ResponseDesc* resp) = 0;

    /**
     * @brief Gets configuration dedicated to plugin behaviour
     * @param name - config key, can be found in ie_plugin_config.hpp
     * @param result - value of config corresponding to config key
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     */
    virtual void GetConfig(const std::string& name, Parameter& result, ResponseDesc* resp) const = 0;

    /**
     * @brief Gets general runtime metric for dedicated hardware
     * @param name  - metric name to request
     * @param result - metric value corresponding to metric key
     * @param resp - Pointer to the response message that holds a description of an error if any
     *             occurred
     */
    virtual void GetMetric(const std::string& name, Parameter& result, ResponseDesc* resp) const = 0;

    /**
     * @brief Gets the remote context.
     * @param pContext  A reference to a context
     * @param resp A response
     */
    virtual void GetContext(RemoteContext::Ptr& pContext, ResponseDesc* resp) const = 0;
};

}  // namespace InferenceEngine
