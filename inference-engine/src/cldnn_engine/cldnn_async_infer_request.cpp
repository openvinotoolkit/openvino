// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_async_infer_request.h"
#include "cldnn_itt.h"
#include <memory>

CLDNNPlugin::CLDNNAsyncInferRequest::CLDNNAsyncInferRequest(const CLDNNInferRequest::Ptr &inferRequest,
                                                            const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                                                            const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                                                            const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor)
    : AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor), _inferRequest(inferRequest), _waitExecutor(waitExecutor) {
    _pipeline = {};

    if (!_inferRequest->use_external_queue()) {
        _pipeline.push_back({taskExecutor,
                    [this] {
                        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNNAsyncInferRequest::PreprocessingAndStartPipeline");
                        _inferRequest->preprocess();
                        _inferRequest->enqueue();
        } });
    }
    _pipeline.push_back({_waitExecutor,
                    [this] {
                        OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNNAsyncInferRequest::WaitPipeline");
                        _inferRequest->wait();
                    }});
}

void CLDNNPlugin::CLDNNAsyncInferRequest::Infer_ThreadUnsafe() {
    if (_inferRequest->use_external_queue()) {
        _inferRequest->preprocess();
        _inferRequest->enqueue();
    }
    Parent::Infer_ThreadUnsafe();
}

void CLDNNPlugin::CLDNNAsyncInferRequest::StartAsync_ThreadUnsafe() {
    if (_inferRequest->use_external_queue()) {
        _inferRequest->preprocess();
        _inferRequest->enqueue();
    }
    Parent::StartAsync_ThreadUnsafe();
}

CLDNNPlugin::CLDNNAsyncInferRequest::~CLDNNAsyncInferRequest() {
    StopAndWait();
}
