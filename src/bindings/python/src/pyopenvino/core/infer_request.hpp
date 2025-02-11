// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include <chrono>
#include <openvino/runtime/infer_request.hpp>

#include "openvino/core/except.hpp"

namespace py = pybind11;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

class InferRequestWrapper {
public:
    // InferRequestWrapper is getting original ov::InferRequest as rvalue.
    // Ownership of the ov::InferRequest is moved to the wrapper
    // while calling upon any of the constructors, so lifetime of
    // the object is managed by the wrapper which is exposed to Python.

    // AsyncInferQueue uses this specifc constructor as setting callback
    // for computing a latency will be done there.
    InferRequestWrapper(ov::InferRequest&& request) : InferRequestWrapper(std::move(request), {}, {}, false) {}

    InferRequestWrapper(ov::InferRequest&& request,
                        const std::vector<ov::Output<const ov::Node>>& inputs,
                        const std::vector<ov::Output<const ov::Node>>& outputs,
                        bool set_default_callback = true,
                        py::object userdata = py::none())
        : m_request{InferRequestWrapper::wrap_infer_request_to_sp(std::move(request))},
          m_inputs{inputs},
          m_outputs{outputs},
          m_userdata{userdata} {
        m_start_time = std::make_shared<Time::time_point>(Time::time_point{});
        m_end_time = std::make_shared<Time::time_point>(Time::time_point{});

        // Initialize InferRequest with default callback
        if (set_default_callback) {
            // Bump reference counter
            auto end_time = m_end_time;
            // Set standard callback which saves "end-time" for inference call
            m_request->set_callback([end_time](std::exception_ptr exception_ptr) {
                *end_time = Time::now();
                try {
                    if (exception_ptr) {
                        std::rethrow_exception(exception_ptr);
                    }
                } catch (const std::exception& e) {
                    OPENVINO_THROW("Caught exception: ", e.what());
                }
            });
        }
    }

    // ~InferRequestWrapper() = default;

    std::vector<ov::Tensor> get_input_tensors() {
        return get_tensors_from(m_inputs);
    }

    std::vector<ov::Tensor> get_output_tensors() {
        return get_tensors_from(m_outputs);
    }

    double get_latency() {
        auto execTime = std::chrono::duration_cast<ns>(*m_end_time - *m_start_time);
        return static_cast<double>(execTime.count()) * 0.000001;
    }

    // Original ov::InferRequest class that is held by this wrapper
    std::shared_ptr<ov::InferRequest> m_request;
    // Inputs and Outputs inherrited from ov::CompiledModel
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;
    // A flag which is set when a user defines a custom callback on InferRequest
    bool m_user_callback_defined = false;
    // Data that is passed by user from Python->C++
    py::object m_userdata;
    // Times of inference's start and finish
    std::shared_ptr<Time::time_point> m_start_time;  // proposal: change to unique_ptr
    std::shared_ptr<Time::time_point> m_end_time;

private:
    inline std::vector<ov::Tensor> get_tensors_from(const std::vector<ov::Output<const ov::Node>>& v) {
        std::vector<ov::Tensor> tensors;
        tensors.reserve(v.size());

        for (auto&& node : v) {
            tensors.push_back(m_request->get_tensor(node));
        }

        return tensors;
    }

    static std::shared_ptr<ov::InferRequest> wrap_infer_request_to_sp(ov::InferRequest request) {
        return std::shared_ptr<ov::InferRequest>(new ov::InferRequest(std::move(request)), [](ov::InferRequest* request) {
                py::gil_scoped_release release;
                delete request;
        });
    }
};

void regclass_InferRequest(py::module m);
