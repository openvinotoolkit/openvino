// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <map>
#include <string>
#include <chrono>

#include "inference_engine.hpp"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

/// @brief Wrapper class for InferenceEngine::InferRequest. Handles asynchronous callbacks and calculates execution time.
class InferReqWrap {
public:
    using Ptr = std::shared_ptr<InferReqWrap>;

    explicit InferReqWrap(InferenceEngine::ExecutableNetwork& net) : _request(net.CreateInferRequest()) {
        _request.SetCompletionCallback(
                [&]() {
                    _endTime = Time::now();
                });
    }

    void startAsync() {
        _startTime = Time::now();
        _request.StartAsync();
    }

    void infer() {
        _startTime = Time::now();
        _request.Infer();
        _endTime = Time::now();
    }

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getPerformanceCounts() {
        return _request.GetPerformanceCounts();
    }

    void wait() {
        InferenceEngine::StatusCode code = _request.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
        if (code != InferenceEngine::StatusCode::OK) {
            throw std::logic_error("Wait");
        }
    }

    InferenceEngine::Blob::Ptr getBlob(const std::string &name) {
        return _request.GetBlob(name);
    }

    double getExecTime() const {
        auto execTime = std::chrono::duration_cast<ns>(_endTime - _startTime);
        return static_cast<double>(execTime.count()) * 0.000001;
    }

private:
    InferenceEngine::InferRequest _request;
    Time::time_point _startTime;
    Time::time_point _endTime;
};