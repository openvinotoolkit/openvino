// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper classes for IExecutableNetwork
 * 
 * @file ie_executable_network.hpp
 */
#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp/ie_cnn_network.h"
#include "cpp/ie_infer_request.hpp"
#include "cpp/ie_memory_state.hpp"
#include "ie_iexecutable_network.hpp"
#include "details/ie_exception_conversion.hpp"
#include "details/ie_so_loader.h"

namespace InferenceEngine {

/**
 * @brief wrapper over IExecutableNetwork
 */
class ExecutableNetwork {
    IExecutableNetwork::Ptr actual;
    details::SharedObjectLoader::Ptr plg;

public:
    /**
     * @brief Default constructor
     */
    ExecutableNetwork() = default;

    /**
     * @brief Destructor
     */
    ~ExecutableNetwork() {
        actual = nullptr;
    }

    /**
     * @brief Constructs ExecutableNetwork from the initialized shared_pointer
     *
     * @param actual Initialized shared pointer
     * @param plg Plugin to use
     */
    explicit ExecutableNetwork(IExecutableNetwork::Ptr actual, details::SharedObjectLoader::Ptr plg = {})
        : actual(actual), plg(plg) {
        //  plg can be null, but not the actual
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "ExecutableNetwork wrapper was not initialized.";
        }
    }

    /**
     * @copybrief IExecutableNetwork::GetOutputsInfo
     *
     * Wraps IExecutableNetwork::GetOutputsInfo.
     * @return A collection that contains string as key, and const Data smart pointer as value
     */
    ConstOutputsDataMap GetOutputsInfo() const {
        ConstOutputsDataMap data;
        CALL_STATUS_FNC(GetOutputsInfo, data);
        return data;
    }

    /**
     * @copybrief IExecutableNetwork::GetInputsInfo
     *
     * Wraps IExecutableNetwork::GetInputsInfo
     * @return A collection that contains string as key, and const InputInfo smart pointer as value
     */
    ConstInputsDataMap GetInputsInfo() const {
        ConstInputsDataMap info;
        CALL_STATUS_FNC(GetInputsInfo, info);
        return info;
    }

    /**
     * @brief reset owned object to new pointer.
     *
     * Eessential for cases when simultaneously loaded networks not expected.
     * @param newActual actual pointed object
     */
    void reset(IExecutableNetwork::Ptr newActual) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "ExecutableNetwork wrapper was not initialized.";
        }
        if (newActual == nullptr) {
            THROW_IE_EXCEPTION << "ExecutableNetwork wrapper used for reset was not initialized.";
        }
        this->actual.swap(newActual);
    }

    /**
     * @copybrief IExecutableNetwork::CreateInferRequest
     *
     * Wraps IExecutableNetwork::CreateInferRequest.
     * @return InferRequest object
     */
    InferRequest CreateInferRequest() {
        IInferRequest::Ptr req;
        CALL_STATUS_FNC(CreateInferRequest, req);
        if (req.get() == nullptr) THROW_IE_EXCEPTION << "Internal error: pointer to infer request is null";
        return InferRequest(req, plg);
    }

    /**
     * @copybrief IExecutableNetwork::CreateInferRequest
     *
     * Wraps IExecutableNetwork::CreateInferRequest.
     * @return shared pointer on InferenceEngine::InferRequest object
     */
    InferRequest::Ptr CreateInferRequestPtr() {
        IInferRequest::Ptr req;
        CALL_STATUS_FNC(CreateInferRequest, req);
        return std::make_shared<InferRequest>(req, plg);
    }

    /**
     * @copybrief IExecutableNetwork::Export
     *
     * Wraps IExecutableNetwork::Export.
     *
     * @see Core::ImportNetwork
     *
     * @param modelFileName Full path to the location of the exported file
     */
    void Export(const std::string& modelFileName) {
        CALL_STATUS_FNC(Export, modelFileName);
    }

    /**
     * @copybrief IExecutableNetwork::Export
     *
     * Wraps IExecutableNetwork::Export.
     *
     * @see Core::ImportNetwork
     *
     * @param networkModel network model output stream
     */
    void Export(std::ostream& networkModel) {
        CALL_STATUS_FNC(Export, networkModel);
    }

    /**
     * @brief cast operator is used when this wrapper initialized by LoadNetwork
     * @return A shared pointer to IExecutableNetwork interface. 
     */
    operator IExecutableNetwork::Ptr&() {
        return actual;
    }

    /**
     * @copybrief IExecutableNetwork::GetExecGraphInfo
     *
     * Wraps IExecutableNetwork::GetExecGraphInfo.
     * @return CNNetwork containing Executable Graph Info
     */
    CNNNetwork GetExecGraphInfo() {
        ICNNNetwork::Ptr ptr = nullptr;
        CALL_STATUS_FNC(GetExecGraphInfo, ptr);
        return CNNNetwork(ptr);
    }

    /**
     * @copybrief IExecutableNetwork::QueryState
     *
     * Wraps IExecutableNetwork::QueryState
     * @return A vector of Memory State objects
     */
    INFERENCE_ENGINE_DEPRECATED("Use InferRequest::QueryState instead")
    std::vector<VariableState> QueryState() {
        if (actual == nullptr) THROW_IE_EXCEPTION << "ExecutableNetwork was not initialized.";
        IVariableState::Ptr pState = nullptr;
        auto res = OK;
        std::vector<VariableState> controller;
        for (size_t idx = 0; res == OK; ++idx) {
            ResponseDesc resp;
            IE_SUPPRESS_DEPRECATED_START
            res = actual->QueryState(pState, idx, &resp);
            IE_SUPPRESS_DEPRECATED_END
            if (res != OK && res != OUT_OF_BOUNDS) {
                THROW_IE_EXCEPTION << resp.msg;
            }
            if (res != OUT_OF_BOUNDS) {
                controller.push_back(VariableState(pState, plg));
            }
        }

        return controller;
    }

    /**
     * @copybrief IExecutableNetwork::SetConfig
     *
     * Wraps IExecutableNetwork::SetConfig.
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    void SetConfig(const std::map<std::string, Parameter>& config) {
        CALL_STATUS_FNC(SetConfig, config);
    }

    /**
     * @copybrief IExecutableNetwork::GetConfig
     *
     * Wraps IExecutableNetwork::GetConfig
     * @param name - config key, can be found in ie_plugin_config.hpp
     * @return Configuration parameter value
     */
    Parameter GetConfig(const std::string& name) const {
        Parameter configValue;
        CALL_STATUS_FNC(GetConfig, name, configValue);
        return configValue;
    }

    /**
     * @copybrief IExecutableNetwork::GetMetric
     *
     * Wraps IExecutableNetwork::GetMetric
     * @param name  - metric name to request
     * @return Metric parameter value
     */
    Parameter GetMetric(const std::string& name) const {
        Parameter metricValue;
        CALL_STATUS_FNC(GetMetric, name, metricValue);
        return metricValue;
    }

    /**
     * @brief Returns pointer to plugin-specific shared context
     * on remote accelerator device that was used to create this ExecutableNetwork
     * @return A context
     */
    RemoteContext::Ptr GetContext() const {
        RemoteContext::Ptr pContext;
        CALL_STATUS_FNC(GetContext, pContext);
        return pContext;
    }

    /**
     * @brief A smart pointer to the ExecutableNetwork object
     */
    using Ptr = std::shared_ptr<ExecutableNetwork>;
};

}  // namespace InferenceEngine
