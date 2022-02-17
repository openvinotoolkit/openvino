// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>

#include <pybind11/pybind11.h>

#include <openvino/runtime/infer_request.hpp>

namespace py = pybind11;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

class InferRequestWrapper {
public:
    InferRequestWrapper(ov::InferRequest request)
        : _request(request)
    {
        // AsyncInferQueue uses this constructor - setting callback for computing a latency will be done there
    }

    InferRequestWrapper(ov::InferRequest request, const std::vector<ov::Output<const ov::Node>>& inputs, const std::vector<ov::Output<const ov::Node>>& outputs)
        : _request(request), _inputs(inputs), _outputs(outputs)
    {
        _request.set_callback([this](std::exception_ptr exception_ptr) {
            _end_time = Time::now();
            try {
                if (exception_ptr) {
                    std::rethrow_exception(exception_ptr);
                }
            } catch (const std::exception& e) {
                throw ov::Exception("Caught exception: " + std::string(e.what()));
            }
        });
    }
    // ~InferRequestWrapper() = default;

    std::vector<ov::Tensor> get_input_tensors() {
        std::vector<ov::Tensor> tensors;
        for (auto&& node : _inputs) {
            tensors.push_back(_request.get_tensor(node));
        }
        return tensors;
    }

    std::vector<ov::Tensor> get_output_tensors() {
        std::vector<ov::Tensor> tensors;
        for (auto&& node : _outputs) {
            tensors.push_back(_request.get_tensor(node));
        }
        return tensors;
    }

    bool user_callback_defined = false;
    py::object userdata;

    double get_latency() {
        auto execTime = std::chrono::duration_cast<ns>(_end_time - _start_time);
        return static_cast<double>(execTime.count()) * 0.000001;
    }

    ov::InferRequest _request;
    std::vector<ov::Output<const ov::Node>> _inputs;
    std::vector<ov::Output<const ov::Node>> _outputs;

    Time::time_point _start_time;
    Time::time_point _end_time;
};

void regclass_InferRequest(py::module m);
