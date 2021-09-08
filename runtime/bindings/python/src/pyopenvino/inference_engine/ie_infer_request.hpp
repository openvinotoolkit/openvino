// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>

#include <pybind11/pybind11.h>

#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_infer_request.hpp>
#include <ie_input_info.hpp>

namespace py = pybind11;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

class InferRequestWrapper {
public:
    InferRequestWrapper(InferenceEngine::InferRequest request)
        : _request(request)
    {
    }
    // ~InferRequestWrapper() = default;

    // bool user_callback_defined;
    // py::function user_callback;

    double getLatency() {
        auto execTime = std::chrono::duration_cast<ns>(_endTime - _startTime);
        return static_cast<double>(execTime.count()) * 0.000001;
    }

    InferenceEngine::InferRequest _request;
    InferenceEngine::ConstInputsDataMap _inputsInfo;
    InferenceEngine::ConstOutputsDataMap _outputsInfo;
    Time::time_point _startTime;
    Time::time_point _endTime;
};

void regclass_InferRequest(py::module m);
