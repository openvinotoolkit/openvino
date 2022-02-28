// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <string>

#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "infer_request.hpp"
#include "executable_network.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
typedef  BaseAsyncInferRequest InferenceEngine::AsyncInferRequestThreadSafeDefault;
class BaseInferRequest : public MultiDeviceInferRequest {
public:
    using Ptr = std::shared_ptr<BaseInferRequest>;

    explicit BaseInferRequest(const InferenceEngine::SoIInferRequestInternal&  inferRequest,
             Schedule::Ptr _schedule);
    ~BaseInferRequest();

    virtual void InferImpl() override;
    InferenceEngine::SoIInferRequestInternal  _realInferRequest;
private:
    Schedule::Ptr                             _schedule;
}  // namespace MultiDevicePlugin

