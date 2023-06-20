// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "sync_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {
class AsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<AsyncInferRequest>;

    explicit AsyncInferRequest(const SyncInferRequest::Ptr& inferRequest,
                                        InferenceEngine::SoIInferRequestInternal& inferRequestWithoutBatch,
                                        const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    void Infer_ThreadUnsafe() override;

    virtual ~AsyncInferRequest();

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    InferenceEngine::SoIInferRequestInternal m_infer_request_without_batch;

    SyncInferRequest::Ptr m_sync_infer_request;
};
}  // namespace autobatch_plugin
}  // namespace ov