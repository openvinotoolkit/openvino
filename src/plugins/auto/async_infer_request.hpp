// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "schedule.hpp"
#include "infer_request.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
class AsyncInferRequest : public IE::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<AsyncInferRequest>;
    explicit AsyncInferRequest(const Schedule::Ptr& schedule, const IInferPtr& inferRequest,
                               const IE::ITaskExecutor::Ptr&  callbackExecutor);
    void Infer_ThreadUnsafe() override;
    std::map<std::string, IE::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    ~AsyncInferRequest();

protected:
    Schedule::Ptr       _schedule;
    WorkerInferRequest* _workerInferRequest = nullptr;
    IInferPtr           _inferRequest;
};

}  // namespace MultiDevicePlugin
