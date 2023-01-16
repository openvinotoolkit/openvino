// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/async_infer_request_legacy.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include <memory>

namespace ov {
namespace intel_gpu {

AsyncInferRequestLegacy::AsyncInferRequestLegacy(const InferRequestLegacy::Ptr &inferRequest,
                                                 const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                                                 const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                                                 const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor)
    : AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor), _inferRequest(inferRequest), _waitExecutor(waitExecutor) {
    _pipeline = {};

    if (!_inferRequest->use_external_queue()) {
        _pipeline.push_back({taskExecutor,
                    [this] {
                        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "AsyncInferRequest::PreprocessingAndStartPipeline");
                        _inferRequest->setup_stream_graph();
                        _inferRequest->preprocess();
                        _inferRequest->enqueue();
                        _inferRequest->wait();
        } });
    } else {
        _pipeline.push_back({ _waitExecutor,
                        [this] {
                            OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "AsyncInferRequest::WaitPipeline");
                            _inferRequest->wait_notify();
                        } });
    }
}

void AsyncInferRequestLegacy::StartAsync_ThreadUnsafe() {
    if (_inferRequest->use_external_queue()) {
        _inferRequest->setup_stream_graph();
        _inferRequest->preprocess_notify();
        _inferRequest->enqueue_notify();
    }
    Parent::StartAsync_ThreadUnsafe();
}

AsyncInferRequestLegacy::~AsyncInferRequestLegacy() {
    StopAndWait();
}

}  // namespace intel_gpu
}  // namespace ov
