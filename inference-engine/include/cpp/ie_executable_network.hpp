// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper classes for IExecutableNetwork
 * @file ie_executable_network.hpp
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include "ie_iexecutable_network.hpp"
#include "ie_plugin_ptr.hpp"
#include "cpp/ie_infer_request.hpp"
#include "cpp/ie_memory_state.hpp"
#include "cpp/ie_cnn_network.h"
#include "details/ie_exception_conversion.hpp"

namespace InferenceEngine {

/**
 * @brief wrapper over IExecutableNetwork
 */
class ExecutableNetwork {
    IExecutableNetwork::Ptr actual;
    InferenceEnginePluginPtr plg;

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
     * @param actual Initialized shared pointer
     * @param plg Plugin to use
     */
    explicit ExecutableNetwork(IExecutableNetwork::Ptr actual, InferenceEnginePluginPtr plg = {})
    : actual(actual), plg(plg) {}

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
    * @see InferencePlugin::ImportNetwork
    * 
    * @param modelFileName Full path to the location of the exported file
    */
    void Export(const std::string &modelFileName) {
        CALL_STATUS_FNC(Export, modelFileName);
    }

    /**
    * @copybrief IExecutableNetwork::GetMappedTopology
    * 
    * Wraps IExecutableNetwork::GetMappedTopology.
    * @param deployedTopology Map of PrimitiveInfo objects that represent the deployed topology
    */
    void GetMappedTopology(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &deployedTopology) {
        CALL_STATUS_FNC(GetMappedTopology, deployedTopology);
    }

    /**
    * cast operator is used when this wrapper initialized by LoadNetwork
    * @return
    */
    operator IExecutableNetwork::Ptr &() {
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
    std::vector<MemoryState> QueryState() {
        IMemoryState::Ptr pState = nullptr;
        auto res = OK;
        std::vector<MemoryState> controller;
        for (size_t idx = 0; res == OK; ++idx) {
            ResponseDesc resp;
            res = actual->QueryState(pState, idx, &resp);
            if (res != OK && res != OUT_OF_BOUNDS) {
                THROW_IE_EXCEPTION << resp.msg;
            }
            if (res != OUT_OF_BOUNDS) {
                controller.push_back(MemoryState(pState));
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
    void SetConfig(const std::map<std::string, Parameter> &config) {
        CALL_STATUS_FNC(SetConfig, config);
    }

    /** @copybrief IExecutableNetwork::GetConfig
     * 
     * Wraps IExecutableNetwork::GetConfig
     * @param name - config key, can be found in ie_plugin_config.hpp
     * @return Configuration paramater value
     */
    Parameter GetConfig(const std::string &name) const {
        Parameter configValue;
        CALL_STATUS_FNC(GetConfig, name, configValue);
        return configValue;
    }

    /**
     * @copybrief IExecutableNetwork::GetMetric
     * 
     * Wraps IExecutableNetwork::GetMetric
     * @param name  - metric name to request
     * @return Metric paramater value
     */
    Parameter GetMetric(const std::string &name) const {
        Parameter metricValue;
        CALL_STATUS_FNC(GetMetric, name, metricValue);
        return metricValue;
    }

    /**
     * @brief A smart pointer to the ExecutableNetwork object
     */
    using Ptr = std::shared_ptr<ExecutableNetwork>;
};

}  // namespace InferenceEngine
