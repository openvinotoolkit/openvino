// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp/ie_executable_network.hpp"
#include "ie_common.h"

namespace InferenceEngine {

ExecutableNetwork::ExecutableNetwork(IExecutableNetwork::Ptr actual_, details::SharedObjectLoader::Ptr plg)
    : actual(actual_), plg(plg) {
    //  plg can be null, but not the actual
    if (actual == nullptr) {
        IE_THROW() << "ExecutableNetwork wrapper was not initialized.";
    }
}

ExecutableNetwork::~ExecutableNetwork() {
    actual = {};
}

ConstOutputsDataMap ExecutableNetwork::GetOutputsInfo() const {
    ConstOutputsDataMap data;
    CALL_STATUS_FNC(GetOutputsInfo, data);
    return data;
}

ConstInputsDataMap ExecutableNetwork::GetInputsInfo() const {
    ConstInputsDataMap info;
    CALL_STATUS_FNC(GetInputsInfo, info);
    return info;
}

void ExecutableNetwork::reset(IExecutableNetwork::Ptr newActual) {
    if (actual == nullptr) {
        IE_THROW() << "ExecutableNetwork wrapper was not initialized.";
    }
    if (newActual == nullptr) {
        IE_THROW() << "ExecutableNetwork wrapper used for reset was not initialized.";
    }
    this->actual.swap(newActual);
}

InferRequest ExecutableNetwork::CreateInferRequest() {
    IInferRequest::Ptr req;
    CALL_STATUS_FNC(CreateInferRequest, req);
    if (req.get() == nullptr) IE_THROW() << "Internal error: pointer to infer request is null";
    return InferRequest(req, plg);
}

InferRequest::Ptr ExecutableNetwork::CreateInferRequestPtr() {
    IInferRequest::Ptr req;
    CALL_STATUS_FNC(CreateInferRequest, req);
    return std::make_shared<InferRequest>(req, plg);
}

void ExecutableNetwork::Export(const std::string& modelFileName) {
    CALL_STATUS_FNC(Export, modelFileName);
}

void ExecutableNetwork::Export(std::ostream& networkModel) {
    CALL_STATUS_FNC(Export, networkModel);
}

ExecutableNetwork::operator IExecutableNetwork::Ptr&() {
    return actual;
}

CNNNetwork ExecutableNetwork::GetExecGraphInfo() {
    IE_SUPPRESS_DEPRECATED_START
    ICNNNetwork::Ptr ptr = nullptr;
    CALL_STATUS_FNC(GetExecGraphInfo, ptr);
    return CNNNetwork(ptr);
    IE_SUPPRESS_DEPRECATED_END
}


std::vector<VariableState> ExecutableNetwork::QueryState() {
    if (actual == nullptr) IE_THROW() << "ExecutableNetwork was not initialized.";
    IVariableState::Ptr pState = nullptr;
    auto res = OK;
    std::vector<VariableState> controller;
    for (size_t idx = 0; res == OK; ++idx) {
        ResponseDesc resp;
        IE_SUPPRESS_DEPRECATED_START
        res = actual->QueryState(pState, idx, &resp);
        IE_SUPPRESS_DEPRECATED_END
        if (res != OK && res != OUT_OF_BOUNDS) {
            IE_THROW() << resp.msg;
        }
        if (res != OUT_OF_BOUNDS) {
            controller.push_back(VariableState(pState, plg));
        }
    }

    return controller;
}

void ExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
    CALL_STATUS_FNC(SetConfig, config);
}

Parameter ExecutableNetwork::GetConfig(const std::string& name) const {
    Parameter configValue;
    CALL_STATUS_FNC(GetConfig, name, configValue);
    return configValue;
}

Parameter ExecutableNetwork::GetMetric(const std::string& name) const {
    Parameter metricValue;
    CALL_STATUS_FNC(GetMetric, name, metricValue);
    return metricValue;
}

RemoteContext::Ptr ExecutableNetwork::GetContext() const {
    RemoteContext::Ptr pContext;
    CALL_STATUS_FNC(GetContext, pContext);
    return pContext;
}
}  // namespace InferenceEngine
