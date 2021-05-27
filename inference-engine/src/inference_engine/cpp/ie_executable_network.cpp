// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_common.h"

#include "cpp/ie_executable_network.hpp"
#include "cpp/exception2status.hpp"
#include "ie_executable_network_base.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"

namespace InferenceEngine {

#define EXEC_NET_CALL_STATEMENT(...)                                                               \
    if (_impl == nullptr) IE_THROW(NotAllocated) << "ExecutableNetwork was not initialized.";      \
    try {                                                                                          \
        __VA_ARGS__;                                                                               \
    } catch(...) {details::Rethrow();}

ExecutableNetwork::ExecutableNetwork(const details::SharedObjectLoader&      so,
                                     const IExecutableNetworkInternal::Ptr&  impl)
    : _so(so), _impl(impl) {
    IE_ASSERT(_impl != nullptr);
}

IE_SUPPRESS_DEPRECATED_START

ExecutableNetwork::ExecutableNetwork(IExecutableNetwork::Ptr exec,
                                     std::shared_ptr<details::SharedObjectLoader> splg)
    : _so(), _impl(), actual(exec) {
    if (splg) {
        _so = *splg;
    }

    //  plg can be null, but not the actual
    if (actual == nullptr)
        IE_THROW(NotAllocated) << "ExecutableNetwork was not initialized.";
}

ConstOutputsDataMap ExecutableNetwork::GetOutputsInfo() const {
    if (actual) {
        ConstOutputsDataMap data;
        CALL_STATUS_FNC(GetOutputsInfo, data);
        return data;
    }

    EXEC_NET_CALL_STATEMENT(return _impl->GetOutputsInfo());
}

ConstInputsDataMap ExecutableNetwork::GetInputsInfo() const {
    if (actual) {
        ConstInputsDataMap info;
        CALL_STATUS_FNC(GetInputsInfo, info);
        return info;
    }

    EXEC_NET_CALL_STATEMENT(return _impl->GetInputsInfo());
}

void ExecutableNetwork::reset(IExecutableNetwork::Ptr newActual) {
    if (actual) {
        if (newActual == nullptr) {
            THROW_IE_EXCEPTION << "ExecutableNetwork wrapper used for reset was not initialized.";
        }
        this->actual.swap(newActual);
        return;
    }

    if (_impl == nullptr) IE_THROW() << "ExecutableNetwork was not initialized.";
    if (newActual == nullptr) IE_THROW() << "ExecutableNetwork wrapper used for reset was not initialized.";
    auto newBase = std::dynamic_pointer_cast<ExecutableNetworkBase>(newActual);
    IE_ASSERT(newBase != nullptr);
    auto newImpl = newBase->GetImpl();
    IE_ASSERT(newImpl != nullptr);
    _impl = newImpl;
}

ExecutableNetwork::operator IExecutableNetwork::Ptr() {
    if (actual) {
        return actual;
    }

    return std::make_shared<ExecutableNetworkBase>(_impl);
}

std::vector<VariableState> ExecutableNetwork::QueryState() {
    if (actual) {
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
                controller.push_back(VariableState(pState,
                    std::make_shared<details::SharedObjectLoader>(_so)));
            }
        }

        return controller;
    }

    std::vector<VariableState> controller;
    EXEC_NET_CALL_STATEMENT(
        for (auto&& state : _impl->QueryState()) {
            controller.emplace_back(VariableState{ _so, state });
        });
    return controller;
}

InferRequest ExecutableNetwork::CreateInferRequest() {
    if (actual) {
        IInferRequest::Ptr req;
        CALL_STATUS_FNC(CreateInferRequest, req);
        if (req.get() == nullptr) THROW_IE_EXCEPTION << "Internal error: pointer to infer request is null";
        return InferRequest(req, std::make_shared<details::SharedObjectLoader>(_so));
    }

    EXEC_NET_CALL_STATEMENT(return {_so, _impl->CreateInferRequest()});
}

InferRequest::Ptr ExecutableNetwork::CreateInferRequestPtr() {
    return std::make_shared<InferRequest>(CreateInferRequest());
}

void ExecutableNetwork::Export(const std::string& modelFileName) {
    if (actual) {
        CALL_STATUS_FNC(Export, modelFileName);
        return;
    }
    EXEC_NET_CALL_STATEMENT(_impl->Export(modelFileName));
}

void ExecutableNetwork::Export(std::ostream& networkModel) {
    if (actual) {
        CALL_STATUS_FNC(Export, networkModel);
        return;
    }
    EXEC_NET_CALL_STATEMENT(_impl->Export(networkModel));
}

CNNNetwork ExecutableNetwork::GetExecGraphInfo() {
    if (actual) {
        IE_SUPPRESS_DEPRECATED_START
        ICNNNetwork::Ptr ptr = nullptr;
        CALL_STATUS_FNC(GetExecGraphInfo, ptr);
        return CNNNetwork(ptr);
        IE_SUPPRESS_DEPRECATED_END
    }
    EXEC_NET_CALL_STATEMENT(return _impl->GetExecGraphInfo());
}

void ExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
    if (actual) {
        CALL_STATUS_FNC(SetConfig, config);
        return;
    }
    EXEC_NET_CALL_STATEMENT(_impl->SetConfig(config));
}

Parameter ExecutableNetwork::GetConfig(const std::string& name) const {
    if (actual) {
        Parameter configValue;
        CALL_STATUS_FNC(GetConfig, name, configValue);
        return configValue;
    }
    EXEC_NET_CALL_STATEMENT(return _impl->GetConfig(name));
}

Parameter ExecutableNetwork::GetMetric(const std::string& name) const {
    if (actual) {
        Parameter metricValue;
        CALL_STATUS_FNC(GetMetric, name, metricValue);
        return metricValue;
    }
    EXEC_NET_CALL_STATEMENT(return _impl->GetMetric(name));
}

RemoteContext::Ptr ExecutableNetwork::GetContext() const {
    if (actual) {
        RemoteContext::Ptr pContext;
        CALL_STATUS_FNC(GetContext, pContext);
        return pContext;
    }
    EXEC_NET_CALL_STATEMENT(return _impl->GetContext());
}

bool ExecutableNetwork::operator!() const noexcept {
    return !_impl || !actual;
}

ExecutableNetwork::operator bool() const noexcept {
    return !!_impl || !!actual;
}
}  // namespace InferenceEngine
