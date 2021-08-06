// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief inference engine executanle network API wrapper, to be used by particular implementors
 * \file ie_executable_network_base.hpp
 */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ie_iexecutable_network.hpp>
#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include "cpp/exception2status.hpp"
#include "ie_infer_async_request_base.hpp"

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START
/**
 * @brief Executable network `noexcept` wrapper which accepts IExecutableNetworkInternal derived instance which can throw exceptions
 * @ingroup ie_dev_api_exec_network_api
  */
class ExecutableNetworkBase : public IExecutableNetwork {
protected:
    std::shared_ptr<IExecutableNetworkInternal> _impl;

public:
    /**
     * @brief Constructor with actual underlying implementation.
     * @param impl Underlying implementation of type IExecutableNetworkInternal
     */
    explicit ExecutableNetworkBase(std::shared_ptr<IExecutableNetworkInternal> impl) {
        if (impl.get() == nullptr) {
            IE_THROW() << "implementation not defined";
        }
        _impl = impl;
    }

    StatusCode GetOutputsInfo(ConstOutputsDataMap& outs, ResponseDesc* resp) const noexcept override {
        TO_STATUS(outs = _impl->GetOutputsInfo());
    }

    StatusCode GetInputsInfo(ConstInputsDataMap& inputs, ResponseDesc* resp) const noexcept override {
        TO_STATUS(inputs = _impl->GetInputsInfo());
    }

    StatusCode CreateInferRequest(IInferRequest::Ptr& req, ResponseDesc* resp) noexcept override {
        TO_STATUS(req = std::make_shared<InferRequestBase>(_impl->CreateInferRequest()));
    }

    StatusCode Export(const std::string& modelFileName, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->Export(modelFileName));
    }

    StatusCode Export(std::ostream& networkModel, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->Export(networkModel));
    }

    StatusCode GetExecGraphInfo(ICNNNetwork::Ptr& graphPtr, ResponseDesc* resp) noexcept override {
        TO_STATUS(graphPtr = _impl->GetExecGraphInfo());
    }

    StatusCode SetConfig(const std::map<std::string, Parameter>& config, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetConfig(config));
    }

    StatusCode GetConfig(const std::string& name, Parameter& result, ResponseDesc* resp) const noexcept override {
        TO_STATUS(result = _impl->GetConfig(name));
    }

    StatusCode GetMetric(const std::string& name, Parameter& result, ResponseDesc* resp) const noexcept override {
        TO_STATUS(result = _impl->GetMetric(name));
    }

    StatusCode GetContext(RemoteContext::Ptr& pContext, ResponseDesc* resp) const noexcept override {
        TO_STATUS(pContext = _impl->GetContext());
    }

    std::shared_ptr<IExecutableNetworkInternal> GetImpl() const {
        return _impl;
    }
};
IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
