// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for IExecutableNetwork interface
 *
 * @file ie_iexecutable_network.hpp
 */
#pragma once

#include <ostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_common.h"
#include "ie_icnn_network.hpp"
#include "ie_iinfer_request.hpp"
#include "ie_input_info.hpp"
#include "ie_parameter.hpp"
#include "ie_remote_context.hpp"

namespace InferenceEngine {

_IE_SUPPRESS_DEPRECATED_START_GCC

/**
 * @brief This is an interface of an executable network
 */
class INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::ExecutableNetwork instead") IExecutableNetwork
    : public std::enable_shared_from_this<IExecutableNetwork> {
public:
    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief A smart pointer to the current IExecutableNetwork object
     */
    using Ptr = std::shared_ptr<IExecutableNetwork>;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Gets the Executable network output Data node information.
     *
     * The received info is stored in the given InferenceEngine::ConstOutputsDataMap node.
     * This method need to be called to find output names for using them later
     * when calling InferenceEngine::InferRequest::GetBlob or InferenceEngine::InferRequest::SetBlob
     *
     * @param out Reference to the InferenceEngine::ConstOutputsDataMap object
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode GetOutputsInfo(ConstOutputsDataMap& out, ResponseDesc* resp) const noexcept = 0;

    /**
     * @brief Gets the executable network input Data node information.
     *
     * The received info is stored in the given InferenceEngine::ConstInputsDataMap object.
     * This method need to be called to find out input names for using them later
     * when calling InferenceEngine::InferRequest::SetBlob
     *
     * @param inputs Reference to InferenceEngine::ConstInputsDataMap object.
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode GetInputsInfo(ConstInputsDataMap& inputs, ResponseDesc* resp) const noexcept = 0;

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief Creates an inference request object used to infer the network.
     *
     * The created request has allocated input and output blobs (that can be changed later).
     *
     * @param req Shared pointer to the created request object
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode CreateInferRequest(IInferRequest::Ptr& req, ResponseDesc* resp) noexcept = 0;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Exports the current executable network.
     *
     * @see Core::ImportNetwork
     *
     * @param modelFileName Full path to the location of the exported file
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode Export(const std::string& modelFileName, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Exports the current executable network.
     *
     * @see Core::ImportNetwork
     *
     * @param networkModel Network model output stream
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    virtual StatusCode Export(std::ostream& networkModel, ResponseDesc* resp) noexcept = 0;

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @deprecated Use InferenceEngine::ExecutableNetwork::GetExecGraphInfo instead
     * @brief Get executable graph information from a device
     *
     * @param graphPtr network ptr to store executable graph information
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: InferenceEngine::OK (0) for success
     */
    INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::ExecutableNetwork::GetExecGraphInfo instead")
    virtual StatusCode GetExecGraphInfo(ICNNNetwork::Ptr& graphPtr, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Sets configuration for current executable network
     *
     * @param config Map of pairs: (config parameter name, config parameter value)
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return code of the operation. InferenceEngine::OK if succeeded
     */
    virtual StatusCode SetConfig(const std::map<std::string, Parameter>& config, ResponseDesc* resp) noexcept = 0;

    /** @brief Gets configuration for current executable network.
     *
     * The method is responsible to extract information
     * which affects executable network execution. The list of supported configuration values can be extracted via
     * ExecutableNetwork::GetMetric with the SUPPORTED_CONFIG_KEYS key, but some of these keys cannot be changed
     * dymanically, e.g. DEVICE_ID cannot changed if an executable network has already been compiled for particular
     * device.
     *
     * @param name config key, can be found in ie_plugin_config.hpp
     * @param result value of config corresponding to config key
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return code of the operation. InferenceEngine::OK if succeeded
     */
    virtual StatusCode GetConfig(const std::string& name, Parameter& result, ResponseDesc* resp) const noexcept = 0;

    /**
     * @brief Gets general runtime metric for an executable network.
     *
     * It can be network name, actual device ID on
     * which executable network is running or all other properties which cannot be changed dynamically.
     *
     * @param name metric name to request
     * @param result metric value corresponding to metric key
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return code of the operation. InferenceEngine::OK if succeeded
     */
    virtual StatusCode GetMetric(const std::string& name, Parameter& result, ResponseDesc* resp) const noexcept = 0;

    /**
     * @brief Gets shared context used to create an executable network.
     *
     * @param pContext Reference to a pointer that will receive resulting shared context object ptr
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return code of the operation. InferenceEngine::OK if succeeded
     */
    virtual StatusCode GetContext(RemoteContext::Ptr& pContext, ResponseDesc* resp) const noexcept = 0;

protected:
    ~IExecutableNetwork() = default;
};

_IE_SUPPRESS_DEPRECATED_END_GCC

}  // namespace InferenceEngine
