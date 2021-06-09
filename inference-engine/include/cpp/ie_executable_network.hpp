// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides ExecutableNetwork class
 *
 * @file ie_executable_network.hpp
 */

#pragma once

#include <ostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_parameter.hpp"
#include "ie_remote_context.hpp"
#include "cpp/ie_cnn_network.h"
#include "cpp/ie_infer_request.hpp"
#include "details/ie_so_loader.h"
#include "ie_iexecutable_network.hpp"

namespace InferenceEngine {
class IExecutableNetworkInternal;

/**
 * @brief This is an interface of an executable network
 */
class INFERENCE_ENGINE_API_CLASS(ExecutableNetwork) {
    details::SharedObjectLoader                  _so;
    std::shared_ptr<IExecutableNetworkInternal>  _impl;
    IE_SUPPRESS_DEPRECATED_START
    std::shared_ptr<IExecutableNetwork>          actual;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Constructs ExecutableNetwork from the initialized std::shared_ptr
     * @param so Plugin to use. This is required to ensure that ExecutableNetwork can work properly even if plugin object is destroyed.
     * @param impl Initialized shared pointer
     */
    ExecutableNetwork(const details::SharedObjectLoader&                   so,
                      const std::shared_ptr<IExecutableNetworkInternal>&   impl);
    friend class Core;

public:
    /**
     * @brief A default constructor.
     */
    ExecutableNetwork() = default;

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @deprecated This ctor will be removed in 2022.1
     * @brief Constructs ExecutableNetwork from the initialized std::shared_ptr
     * @param exec Initialized shared pointer
     * @param splg Plugin to use. This is required to ensure that ExecutableNetwork can work properly even if plugin object is destroyed.
     */
    INFERENCE_ENGINE_DEPRECATED("This ctor will be removed in 2022.1")
    explicit ExecutableNetwork(std::shared_ptr<IExecutableNetwork> exec,
                               std::shared_ptr<details::SharedObjectLoader> splg = {});
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Gets the Executable network output Data node information.
     *
     * The received info is stored in the given InferenceEngine::ConstOutputsDataMap node.
     * This method need to be called to find output names for using them later
     * when calling InferenceEngine::InferRequest::GetBlob or InferenceEngine::InferRequest::SetBlob
     *
     * @return A collection that contains string as key, and const Data smart pointer as value
     */
    ConstOutputsDataMap GetOutputsInfo() const;

    /**
     * @brief Gets the executable network input Data node information.
     *
     * The received info is stored in the given InferenceEngine::ConstInputsDataMap object.
     * This method need to be called to find out input names for using them later
     * when calling InferenceEngine::InferRequest::SetBlob
     *
     * @param inputs Reference to InferenceEngine::ConstInputsDataMap object.
     * @return A collection that contains string as key, and const InputInfo smart pointer as value
     */
    ConstInputsDataMap GetInputsInfo() const;

    /**
     * @brief Creates an inference request object used to infer the network.
     *
     * The created request has allocated input and output blobs (that can be changed later).
     *
     * @return InferRequest object
     */
    InferRequest CreateInferRequest();

    /**
     * @brief Exports the current executable network.
     *
     * @see Core::ImportNetwork
     *
     * @param modelFileName Full path to the location of the exported file
     */
    void Export(const std::string& modelFileName);

    /**
     * @brief Exports the current executable network.
     *
     * @see Core::ImportNetwork
     *
     * @param networkModel Network model output stream
     */
    void Export(std::ostream& networkModel);

    /**
     * @copybrief IExecutableNetwork::GetExecGraphInfo
     *
     * Wraps IExecutableNetwork::GetExecGraphInfo.
     * @return CNNetwork containing Executable Graph Info
     */
    CNNNetwork GetExecGraphInfo();

    /**
     * @brief Sets configuration for current executable network
     *
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    void SetConfig(const std::map<std::string, Parameter>& config);

    /** @brief Gets configuration for current executable network.
     *
     * The method is responsible to extract information
     * which affects executable network execution. The list of supported configuration values can be extracted via
     * ExecutableNetwork::GetMetric with the SUPPORTED_CONFIG_KEYS key, but some of these keys cannot be changed
     * dynamically, e.g. DEVICE_ID cannot changed if an executable network has already been compiled for particular
     * device.
     *
     * @param name config key, can be found in ie_plugin_config.hpp
     * @return Configuration parameter value
     */
    Parameter GetConfig(const std::string& name) const;

    /**
     * @brief Gets general runtime metric for an executable network.
     *
     * It can be network name, actual device ID on
     * which executable network is running or all other properties which cannot be changed dynamically.
     *
     * @param name metric name to request
     * @return Metric parameter value
     */
    Parameter GetMetric(const std::string& name) const;

    /**
     * @brief Returns pointer to plugin-specific shared context
     * on remote accelerator device that was used to create this ExecutableNetwork
     * @return A context
     */
    RemoteContext::Ptr GetContext() const;

    /**
     * @brief Checks if current ExecutableNetwork object is not initialized
     * @return true if current ExecutableNetwork object is not initialized, false - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current ExecutableNetwork object is initialized
     * @return true if current ExecutableNetwork object is initialized, false - otherwise
     */
    explicit operator bool() const noexcept;

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @deprecated The method Will be removed
     * @brief reset owned object to new pointer.
     *
     * Essential for cases when simultaneously loaded networks not expected.
     * @param newActual actual pointed object
     */
    INFERENCE_ENGINE_DEPRECATED("The method will be removed")
    void reset(std::shared_ptr<IExecutableNetwork> newActual);

    /**
     * @deprecated Will be removed. Use operator bool
     * @brief cast operator is used when this wrapper initialized by LoadNetwork
     * @return A shared pointer to IExecutableNetwork interface.
     */
    INFERENCE_ENGINE_DEPRECATED("The method will be removed. Use operator bool")
    operator std::shared_ptr<IExecutableNetwork>();

    /**
     * @deprecated Use ExecutableNetwork::CreateInferRequest
     * @copybrief IExecutableNetwork::CreateInferRequest
     *
     * Wraps IExecutableNetwork::CreateInferRequest.
     * @return shared pointer on InferenceEngine::InferRequest object
     */
    INFERENCE_ENGINE_DEPRECATED("Use ExecutableNetwork::CreateInferRequest instead")
    InferRequest::Ptr CreateInferRequestPtr();

    /**
     * @deprecated Use InferRequest::QueryState instead
     * @brief Gets state control interface for given executable network.
     *
     * State control essential for recurrent networks
     *
     * @return A vector of Memory State objects
     */
    INFERENCE_ENGINE_DEPRECATED("Use InferRequest::QueryState instead")
    std::vector<VariableState> QueryState();
    IE_SUPPRESS_DEPRECATED_END
};

}  // namespace InferenceEngine
