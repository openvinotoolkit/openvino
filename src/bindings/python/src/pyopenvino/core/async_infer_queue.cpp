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
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return !(_idle_handles.empty());
        });
        if (_errors.size() > 0)
            throw _errors.front();
        return !(_idle_handles.empty());
    }

    size_t get_idle_request_id() {
        // Wait for any of _idle_handles
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return !(_idle_handles.empty());
        });
        if (_errors.size() > 0)
            throw _errors.front();
        return _idle_handles.front();
    }

    void wait_all() {
        // Wait for all requests to return with callback thus updating
        // _idle_handles so it matches the size of requests
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return _idle_handles.size() == _requests.size();
        });
        if (_errors.size() > 0)
            throw _errors.front();
    }

    void set_default_callbacks() {
        for (size_t handle = 0; handle < _requests.size(); handle++) {
            _requests[handle]._request.set_callback([this, handle /* ... */](std::exception_ptr exception_ptr) {
                _requests[handle]._end_time = Time::now();
                // Add idle handle to queue
                _idle_handles.push(handle);
                // Notify locks in getIdleRequestId() or waitAll() functions
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
                    _errors.push(py_error);
                }
                // Add idle handle to queue
                _idle_handles.push(handle);
                // Notify locks in getIdleRequestId() or waitAll() functions
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
    cls.doc() = "openvino.runtime.AsyncInferQueue represents helper that creates a pool of asynchronous"
                "InferRequests and provides synchronization functions to control flow of a simple pipeline.";

    cls.def(py::init([](ov::CompiledModel& model, size_t jobs) {
                if (jobs == 0) {
                    jobs = (size_t)Common::get_optimal_number_of_requests(model);
                }

                std::vector<InferRequestWrapper> requests;
                std::queue<size_t> idle_handles;
                std::vector<py::object> user_ids(jobs);

                for (size_t handle = 0; handle < jobs; handle++) {
                    auto request = InferRequestWrapper(model.create_infer_request());
                    // Get Inputs and Outputs info from compiled model
                    request._inputs = model.inputs();
                    request._outputs = model.outputs();

                    requests.push_back(request);
                    idle_handles.push(handle);
                }

                return new AsyncInferQueue(requests, idle_handles, user_ids);
            }),
            py::arg("model"),
            py::arg("jobs") = 0,
            R"(
                Creates AsyncInferQueue.

                Parameters
                ----------
                model : openvino.runtime.CompiledModel
                    Model to be used as a base for a pool of InferRequests.

                jobs : int
                    Number of InferRequests objects in a pool. If 0, jobs number
                    will be set automatically to the optimal number.
                    Default: 0

                Returns
                ----------
                __init__ : openvino.runtime.AsyncInferQueue
            )");

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
        py::arg("userdata"),
        R"(
            Run asynchronous inference using next available InferRequest.

            Parameters
            ----------
            inputs : dict[Union[int, str, openvino.runtime.ConstOutput] : openvino.runtime.Tensor]
                Data to set on input tensors of next available InferRequest from
                AsyncInferQueue's pool.

            userdata : Any
                Any data that will be passed to a callback.

            Returns
            ----------
            start_async : None
        )");

    cls.def(
        "is_ready",
        [](AsyncInferQueue& self) {
            return self._is_ready();
        },
        R"(
            One of 'flow control' functions. Blocking call.
            If there is at least one free InferRequest in a pool, returns True. 

            Parameters
            ----------
            None

            Returns
            ----------
            is_ready : bool
    )");

    cls.def(
        "wait_all",
        [](AsyncInferQueue& self) {
            return self.wait_all();
        },
        R"(
        One of 'flow control' functions. Blocking call.
        Waits for all InferRequests in a pool to finish scheduled work. 

        Parameters
        ----------
        None

        Returns
        ----------
        wait_all : None
    )");

    cls.def(
        "get_idle_request_id",
        [](AsyncInferQueue& self) {
            return self.get_idle_request_id();
        },
        R"(
        Returns next free id of InferRequest from queue's pool.

        Parameters
        ----------
        None

        Returns
        ----------
        get_idle_request_id : int
    )");

    cls.def(
        "set_callback",
        [](AsyncInferQueue& self, py::function callback) {
            self.set_custom_callbacks(callback);
        },
        R"(
        Sets unified callback on all InferRequests from queue's pool.
        Signature of such function should have two arguments, where
        first one is InferRequest object and second one is userdata
        connected to InferRequest from the AsyncInferQueue's pool.

        Example:

            def f(request, userdata):
                result = request.output_tensors[0]
                print(result + userdata)

            async_infer_queue.set_callback(f)

        Parameters
        ----------
        callback : function
            Any Python defined function that matches callback's requirements.

        Returns
        ----------
        set_callback : None
    )");

    cls.def(
        "__len__",
        [](AsyncInferQueue& self) {
            return self._requests.size();
        },
        R"(
        Returns
        ----------
        __len__ : int
            Number of InferRequests in the pool.
    )");

    cls.def(
        "__iter__",
        [](AsyncInferQueue& self) {
            return py::make_iterator(self._requests.begin(), self._requests.end());
        },
        py::keep_alive<0, 1>()); /* Keep set alive while iterator is used */

    cls.def(
        "__getitem__",
        [](AsyncInferQueue& self, size_t i) {
            return self._requests[i];
        },
        R"(
        Parameters
        ----------
        i : int
            InferRequest id. 

        Returns
        ----------
        __getitem__ : openvino.runtime.InferRequest
            InferRequests from the pool with given id.
    )");

    cls.def_property_readonly(
        "userdata",
        [](AsyncInferQueue& self) {
            return self._user_ids;
        },
        R"(
        Returns
        ----------
        userdata : list[Any]
            List of all passed userdata. None if the data wasn't passed yet.
    )");
}
