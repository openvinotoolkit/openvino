// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pyopenvino/core/async_infer_queue.hpp"

#include <ie_common.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/infer_request.hpp"

namespace py = pybind11;

class AsyncInferQueue {
public:
    AsyncInferQueue(std::vector<InferRequestWrapper> requests,
                    std::queue<size_t> idle_handles,
                    std::vector<py::object> user_ids)
        : _requests(requests),
          _idle_handles(idle_handles),
          _user_ids(user_ids) {
        this->set_default_callbacks();
    }

    ~AsyncInferQueue() {
        _requests.clear();
    }

    bool _is_ready() {
        // Check if any request has finished already
        py::gil_scoped_release release;
        // acquire the mutex to access _errors and _idle_handles
        std::lock_guard<std::mutex> lock(_mutex);
        if (_errors.size() > 0)
            throw _errors.front();
        return !(_idle_handles.empty());
    }

    size_t get_idle_request_id() {
        // Wait for any request to complete and return its id
        // release GIL to avoid deadlock on python callback
        py::gil_scoped_release release;
        // acquire the mutex to access _errors and _idle_handles
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return !(_idle_handles.empty());
        });
        size_t idle_handle = _idle_handles.front();
        // wait for request to make sure it returned from callback
        _requests[idle_handle]._request.wait();
        if (_errors.size() > 0)
            throw _errors.front();
        return idle_handle;
    }

    void wait_all() {
        // Wait for all request to complete
        // release GIL to avoid deadlock on python callback
        py::gil_scoped_release release;
        for (auto&& request : _requests) {
            request._request.wait();
        }
        // acquire the mutex to access _errors
        std::lock_guard<std::mutex> lock(_mutex);
        if (_errors.size() > 0)
            throw _errors.front();
    }

    void set_default_callbacks() {
        for (size_t handle = 0; handle < _requests.size(); handle++) {
            _requests[handle]._request.set_callback([this, handle /* ... */](std::exception_ptr exception_ptr) {
                _requests[handle]._end_time = Time::now();
                try {
                    if (exception_ptr) {
                        std::rethrow_exception(exception_ptr);
                    }
                } catch (const std::exception& e) {
                    throw ov::Exception(e.what());
                }
                {
                    // acquire the mutex to access _idle_handles
                    std::lock_guard<std::mutex> lock(_mutex);
                    // Add idle handle to queue
                    _idle_handles.push(handle);
                }
                // Notify locks in getIdleRequestId()
                _cv.notify_one();
            });
        }
    }

    void set_custom_callbacks(py::function f_callback) {
        for (size_t handle = 0; handle < _requests.size(); handle++) {
            _requests[handle]._request.set_callback([this, f_callback, handle](std::exception_ptr exception_ptr) {
                _requests[handle]._end_time = Time::now();
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
                    f_callback(_requests[handle], _user_ids[handle]);
                } catch (py::error_already_set py_error) {
                    assert(PyErr_Occurred());
                    // acquire the mutex to access _errors
                    std::lock_guard<std::mutex> lock(_mutex);
                    _errors.push(py_error);
                }
                {
                    // acquire the mutex to access _idle_handles
                    std::lock_guard<std::mutex> lock(_mutex);
                    // Add idle handle to queue
                    _idle_handles.push(handle);
                }
                // Notify locks in getIdleRequestId()
                _cv.notify_one();
            });
        }
    }

    std::vector<InferRequestWrapper> _requests;
    std::queue<size_t> _idle_handles;
    std::vector<py::object> _user_ids;  // user ID can be any Python object
    std::mutex _mutex;
    std::condition_variable _cv;
    std::queue<py::error_already_set> _errors;
};

void regclass_AsyncInferQueue(py::module m) {
    py::class_<AsyncInferQueue, std::shared_ptr<AsyncInferQueue>> cls(m, "AsyncInferQueue");

    cls.def(py::init([](ov::CompiledModel& net, size_t jobs) {
                if (jobs == 0) {
                    jobs = (size_t)Common::get_optimal_number_of_requests(net);
                }

                std::vector<InferRequestWrapper> requests;
                std::queue<size_t> idle_handles;
                std::vector<py::object> user_ids(jobs);

                for (size_t handle = 0; handle < jobs; handle++) {
                    auto request = InferRequestWrapper(net.create_infer_request());
                    // Get Inputs and Outputs info from executable network
                    request._inputs = net.inputs();
                    request._outputs = net.outputs();

                    requests.push_back(request);
                    idle_handles.push(handle);
                }

                return new AsyncInferQueue(requests, idle_handles, user_ids);
            }),
            py::arg("network"),
            py::arg("jobs") = 0);

    cls.def(
        "start_async",
        [](AsyncInferQueue& self, const py::dict inputs, py::object userdata) {
            // getIdleRequestId function has an intention to block InferQueue
            // until there is at least one idle (free to use) InferRequest
            auto handle = self.get_idle_request_id();
            self._idle_handles.pop();
            // Set new inputs label/id from user
            self._user_ids[handle] = userdata;
            // Update inputs if there are any
            Common::set_request_tensors(self._requests[handle]._request, inputs);
            // Now GIL can be released - we are NOT working with Python objects in this block
            {
                py::gil_scoped_release release;
                self._requests[handle]._start_time = Time::now();
                // Start InferRequest in asynchronus mode
                self._requests[handle]._request.start_async();
            }
        },
        py::arg("inputs"),
        py::arg("userdata"));

    cls.def("is_ready", [](AsyncInferQueue& self) {
        return self._is_ready();
    });

    cls.def("wait_all", [](AsyncInferQueue& self) {
        return self.wait_all();
    });

    cls.def("get_idle_request_id", [](AsyncInferQueue& self) {
        return self.get_idle_request_id();
    });

    cls.def("set_callback", [](AsyncInferQueue& self, py::function f_callback) {
        self.set_custom_callbacks(f_callback);
    });

    cls.def("__len__", [](AsyncInferQueue& self) {
        return self._requests.size();
    });

    cls.def(
        "__iter__",
        [](AsyncInferQueue& self) {
            return py::make_iterator(self._requests.begin(), self._requests.end());
        },
        py::keep_alive<0, 1>()); /* Keep set alive while iterator is used */

    cls.def("__getitem__", [](AsyncInferQueue& self, size_t i) {
        return self._requests[i];
    });

    cls.def_property_readonly("userdata", [](AsyncInferQueue& self) {
        return self._user_ids;
    });
}
