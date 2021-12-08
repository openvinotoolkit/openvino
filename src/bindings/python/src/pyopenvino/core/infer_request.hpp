// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>

#include <pybind11/pybind11.h>

#include <openvino/runtime/infer_request.hpp>

namespace py = pybind11;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

class PyInferRequest {
public:
    PyInferRequest(ov::runtime::InferRequest request, const std::vector<ov::Output<const ov::Node>>& inputs, const std::vector<ov::Output<const ov::Node>>& outputs)
        : cpp_request(request), _inputs(inputs), _outputs(outputs)
    {
        cpp_request.set_callback([this](std::exception_ptr exception_ptr) {
            _end_time = Time::now();
        });
    }

    double get_latency() {
        auto execTime = std::chrono::duration_cast<ns>(_end_time - _start_time);
        return static_cast<double>(execTime.count()) * 0.000001;
    }

    ov::runtime::InferRequest cpp_request;

    std::vector<ov::Output<const ov::Node>> _inputs;
    std::vector<ov::Output<const ov::Node>> _outputs;

    Time::time_point _start_time;
    Time::time_point _end_time;

    bool user_callback_defined = false;
    py::object userdata;
};

void regclass_InferRequest(py::module m);
