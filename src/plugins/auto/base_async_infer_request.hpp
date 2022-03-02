// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once


#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "infer_request.hpp"
#include "base_schedule.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
class BaseAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<BaseAsyncInferRequest>;
    explicit BaseAsyncInferRequest(const Schedule::Ptr& schedule,
            const IInferPtr&         inferRequest,
            const InferenceEngine::ITaskExecutor::Ptr&  callbackExecutor);
    void Infer_ThreadUnsafe() override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    ~BaseAsyncInferRequest();

protected:
    Schedule::Ptr _schedule;
    WorkerInferRequest* _workerInferRequest = nullptr;
    IInferPtr _inferRequest;
};

}  // namespace MultiDevicePlugin
