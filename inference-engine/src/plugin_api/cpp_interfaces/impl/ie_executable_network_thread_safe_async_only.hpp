// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iinfer_request.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp_interfaces/base/ie_infer_async_request_base.hpp"
#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_async_request_internal.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp"

namespace InferenceEngine {

/**
 * @brief      This class describes an executable network thread safe asynchronous only implementation.
 * @ingroup    ie_dev_api_exec_network_api
 */
class ExecutableNetworkThreadSafeAsyncOnly : public ExecutableNetworkInternal,
                                             public std::enable_shared_from_this<ExecutableNetworkThreadSafeAsyncOnly> {
public:
    /**
     * @brief A shared pointer to a ExecutableNetworkThreadSafeAsyncOnly object
     */
    typedef std::shared_ptr<ExecutableNetworkThreadSafeAsyncOnly> Ptr;

    /**
     * @brief      Creates an asynchronous inference request public implementation.
     * @param      asyncRequest  The asynchronous request public implementation
     */
    void CreateInferRequest(IInferRequest::Ptr& asyncRequest) override {
        auto asyncRequestImpl = this->CreateAsyncInferRequestImpl(_networkInputs, _networkOutputs);
        asyncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        asyncRequest.reset(new InferRequestBase<AsyncInferRequestInternal>(asyncRequestImpl), [](IInferRequest* p) {
            p->Release();
        });
        asyncRequestImpl->SetPublicInterfacePtr(asyncRequest);
    }

protected:
    /**
     * @brief      Creates an asynchronous inference request internal implementation.
     * @note       The method is called by ExecutableNetworkThreadSafeAsyncOnly::CreateInferRequest as
     *             plugin-specific implementation.
     * @param[in]  networkInputs   The network inputs
     * @param[in]  networkOutputs  The network outputs
     * @return     A shared pointer to asynchnous inference request object.
     */
    virtual AsyncInferRequestInternal::Ptr CreateAsyncInferRequestImpl(InputsDataMap networkInputs,
                                                                       OutputsDataMap networkOutputs) = 0;
};

}  // namespace InferenceEngine
