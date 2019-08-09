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
    ExecutableNetwork() = default;
    ~ExecutableNetwork() {
        actual = nullptr;
    }

    explicit ExecutableNetwork(IExecutableNetwork::Ptr actual, InferenceEnginePluginPtr plg = {})
    : actual(actual), plg(plg) {}

    /**
     * @brief Wraps original method
     * IExecutableNetwork::getOutputsInfo
     */
    ConstOutputsDataMap GetOutputsInfo() const {
        ConstOutputsDataMap data;
        CALL_STATUS_FNC(GetOutputsInfo, data);
        return data;
    }

    /**
     * @brief Wraps original method
     * IExecutableNetwork::getInputsInfo
     */
    ConstInputsDataMap GetInputsInfo() const {
        ConstInputsDataMap info;
        CALL_STATUS_FNC(GetInputsInfo, info);
        return info;
    }

    /**
     * @brief reset owned object to new pointer, essential for cases when simultaneously loaded networks not expected
     * @param actual actual pointed object
     */
    void reset(IExecutableNetwork::Ptr newActual) {
        this->actual.swap(newActual);
    }

    /**
     * @brief Wraps original method
     * IExecutableNetwork::CreateInferRequest
     */
    InferRequest CreateInferRequest() {
        IInferRequest::Ptr req;
        CALL_STATUS_FNC(CreateInferRequest, req);
        if (req.get() == nullptr) THROW_IE_EXCEPTION << "Internal error: pointer to infer request is null";
        return InferRequest(req, plg);
    }

    /**
     * @brief Wraps original method
     * IExecutableNetwork::CreateInferRequestPtr
     * @return shared pointer on InferRequest object
     */
    InferRequest::Ptr CreateInferRequestPtr() {
        IInferRequest::Ptr req;
        CALL_STATUS_FNC(CreateInferRequest, req);
        return std::make_shared<InferRequest>(req, plg);
    }

    /**
    * @brief Exports the current executable network so it can be used later in the Import() main API
    * @param modelFileName Full path to the location of the exported file
    * @param resp Optional: pointer to an already allocated object to contain information in case of failure
    */
    void Export(const std::string &modelFileName) {
        CALL_STATUS_FNC(Export, modelFileName);
    }

    /**
    * @brief Gets the mapping of IR layer names to implemented kernels
    * @param deployedTopology Map of PrimitiveInfo objects that represent the deployed topology
    * @param resp Optional: pointer to an already allocated object to contain information in case of failure
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
    * @brief Get executable graph information from a plugin represented as CNNNetwork
    * @return CNNetwork containing Executable Graph Info
    */
    CNNNetwork GetExecGraphInfo() {
        ICNNNetwork::Ptr ptr = nullptr;
        CALL_STATUS_FNC(GetExecGraphInfo, ptr);
        return CNNNetwork(ptr);
    }

    /**
     *@brief see original function InferenceEngine::IExecutableNetwork::QueryState
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
     * @brief Sets configuration for current executable network
     * @param config Map of pairs: (config parameter name, config parameter value)
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     */
    void SetConfig(const std::map<std::string, Parameter> &config) {
        CALL_STATUS_FNC(SetConfig, config);
    }

    /** @brief Gets configuration dedicated to plugin behaviour
        * @param name - config key, can be found in ie_plugin_config.hpp
        * @param options - configuration details for coonfig value
        * @param result - value of config corresponding to config key
        * @param resp Pointer to the response message that holds a description of an error if any occurred
    */
    Parameter GetConfig(const std::string &name) const {
        Parameter configValue;
        CALL_STATUS_FNC(GetConfig, name, configValue);
        return configValue;
    }

    /**
     * @brief Gets general runtime metric for dedicated hardware
     * @param name  - metric name to request
     * @param options - configuration details for metric
     * @param result - metric value corresponding to metric key
     * @param resp - Pointer to the response message that holds a description of an error if any
     *             occurred
     * @return code of the operation. OK if succeeded
     */
    Parameter GetMetric(const std::string &name) const {
        Parameter metricValue;
        CALL_STATUS_FNC(GetMetric, name, metricValue);
        return metricValue;
    }

    using Ptr = std::shared_ptr<ExecutableNetwork>;
};

}  // namespace InferenceEngine
