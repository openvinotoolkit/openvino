// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "myriad_infer_request.h"

namespace vpu {
namespace MyriadPlugin {

class MyriadAsyncInferRequest : virtual public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    MyriadAsyncInferRequest(MyriadInferRequest::Ptr request,
                                const InferenceEngine::ITaskExecutor::Ptr &taskExecutorStart,
                                const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor,
                                const InferenceEngine::ITaskExecutor::Ptr &taskExecutorGetResult);

    ~MyriadAsyncInferRequest() override;
private:
    MyriadInferRequest::Ptr _request;
    InferenceEngine::ITaskExecutor::Ptr _taskExecutorGetResult;
};

}  // namespace MyriadPlugin
}  // namespace vpu
