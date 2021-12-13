// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/runtime/async_infer_queue.hpp"

#include <ie_common.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "pyopenvino/core/async_infer_queue.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/infer_request.hpp"

namespace py = pybind11;

struct PyAsyncInferQueue {
public:
    PyAsyncInferQueue(ov::runtime::CompiledModel& model, size_t jobs) {
        if (jobs == 0) {
            jobs = helpers::num_of_jobs_helper(model);
        }

        std::queue<size_t> handles;
        std::vector<ov::Any> userdata;

        // Populate vectors with data.
        // NOTE: Userdata has to be initialized with nullptrs to comply with
        // AsyncInferQueue internal implementation standards!
        for (size_t handle = 0; handle < jobs; handle++) {
            m_pypool.push_back(std::pair<PyInferRequest, py::object>(
                PyInferRequest(model.create_infer_request(), model.inputs(), model.outputs()),
                py::none()));
            handles.push(handle);
            userdata.push_back(ov::Any(nullptr));
        }

        // Vector of references that will be used internally by AsyncInferQueue.
        std::vector<std::reference_wrapper<ov::runtime::InferRequest>> ref_pool;

        for (auto& job : m_pypool) {
            ref_pool.push_back(std::reference_wrapper<ov::runtime::InferRequest>(job.first.cpp_request));
        }

        // Create AsyncInferQueue using ctor that accepts "outside" references.
        m_queue = ov::runtime::AsyncInferQueue(std::move(ref_pool), std::move(handles), std::move(userdata));

        // Callbacks are required to be overwriten with new default ones to obtain
        // latency and throw cpp errors back to Python interpreter.
        set_py_default_callback();
    }

    bool is_ready() {
        py::gil_scoped_release release;
        auto ready_flag = m_queue.is_ready();
        if (m_pyerrors.size() > 0)
            throw m_pyerrors.front();
        return ready_flag;
    }

    size_t get_idle_handle() {
        // Wait for any of _idle_handles
        py::gil_scoped_release release;
        auto handle = m_queue.get_idle_handle();
        if (m_pyerrors.size() > 0)
            throw m_pyerrors.front();
        return handle;
    }

    void wait_all() {
        py::gil_scoped_release release;
        m_queue.wait_all();
        if (m_pyerrors.size() > 0)
            throw m_pyerrors.front();
    }

    void set_py_default_callback() {
        for (size_t handle = 0; handle < m_queue.size(); handle++) {
            m_queue.set_job_callback(handle,
                                     [this, handle](std::exception_ptr exception_ptr,
                                                    ov::runtime::InferRequest& request,
                                                    const ov::Any& userdata) {
                                         m_pypool[handle].first._end_time = Time::now();
                                         try {
                                             if (exception_ptr) {
                                                 std::rethrow_exception(exception_ptr);
                                             }
                                         } catch (const std::exception& e) {
                                             throw ov::Exception(e.what());
                                         }
                                     });
        }
    }

    void set_py_custom_callback(py::function f_callback) {
        for (size_t handle = 0; handle < m_queue.size(); handle++) {
            m_queue.set_job_callback(handle,
                                     [this, f_callback, handle](std::exception_ptr exception_ptr,
                                                                ov::runtime::InferRequest& request,
                                                                const ov::Any& userdata) {
                                         m_pypool[handle].first._end_time = Time::now();
                                         try {
                                             if (exception_ptr) {
                                                 std::rethrow_exception(exception_ptr);
                                             }
                                         } catch (const std::exception& e) {
                                             throw ov::Exception(e.what());
                                         }
                                         // Acquire GIL, execute Python function
                                         py::gil_scoped_acquire acquire;
                                         try {
                                             f_callback(m_pypool[handle].first, m_pypool[handle].second);
                                         } catch (py::error_already_set py_error) {
                                             assert(PyErr_Occurred());
                                             m_pyerrors.push(py_error);
                                         }
                                     });
        }
    }

    // Underlaying C++ AsyncInferQueue.
    ov::runtime::AsyncInferQueue m_queue;
    // This pool differs from original pool from AsyncInferQueue.
    // It holds pairs of Python Wrappers for InferRequest and py::objects
    // which are used as userdata inside Python-based callbacks.
    std::vector<std::pair<PyInferRequest, py::object>> m_pypool;
    // Special vector to hold Python errors thrown inside callbacks.
    // TO-DO: investigate other solutions to the problem.
    std::queue<py::error_already_set> m_pyerrors;
};

void regclass_AsyncInferQueue(py::module m) {
    py::class_<PyAsyncInferQueue, std::shared_ptr<PyAsyncInferQueue>> cls(m, "AsyncInferQueue");

    cls.def(py::init<ov::runtime::CompiledModel&, size_t>(), py::arg("model"), py::arg("jobs") = 0);

    cls.def(
        "start_async",
        [](PyAsyncInferQueue& self, py::object& userdata) {
            auto handle = self.m_queue.get_idle_handle();
            self.m_pypool[handle].second = userdata;
            {
                py::gil_scoped_release release;
                self.m_pypool[handle].first._start_time = Time::now();
                // Start InferRequest in asynchronus mode
                self.m_queue.start_async();
            }
        },
        py::arg("userdata"));

    cls.def(
        "start_async",
        [](PyAsyncInferQueue& self, const py::dict& inputs, py::object& userdata) {
            auto handle = self.get_idle_handle();
            Common::set_request_tensors(self.m_queue[handle], inputs);
            self.m_pypool[handle].second = userdata;
            {
                py::gil_scoped_release release;
                self.m_pypool[handle].first._start_time = Time::now();
                // Start InferRequest in asynchronus mode
                self.m_queue.start_async();
            }
        },
        py::arg("inputs"),
        py::arg("userdata"));

    cls.def("is_ready", [](PyAsyncInferQueue& self) {
        return self.is_ready();
    });

    cls.def("wait_all", [](PyAsyncInferQueue& self) {
        return self.wait_all();
    });

    cls.def("get_idle_handle", [](PyAsyncInferQueue& self) {
        return self.get_idle_handle();
    });

    cls.def("set_callback", [](PyAsyncInferQueue& self, py::function f_callback) {
        self.set_py_custom_callback(f_callback);
    });

    cls.def("size", [](PyAsyncInferQueue& self) {
        return self.m_pypool.size();  // or get_jobs_number() ?
    });

    cls.def("__len__", [](PyAsyncInferQueue& self) {
        return self.m_pypool.size();  // or get_jobs_number() ?
    });

    cls.def(
        "__iter__",
        [](PyAsyncInferQueue& self) {
            return py::make_iterator(self.m_pypool.begin(), self.m_pypool.end());
        },
        py::keep_alive<0, 1>()); /* Keep set alive while iterator is used */

    cls.def("__getitem__", [](PyAsyncInferQueue& self, size_t i) {
        return self.m_pypool[i].first;
    });
}
