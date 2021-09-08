// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pyopenvino/core/ie_infer_queue.hpp"

#include <ie_common.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <chrono>
#include <condition_variable>
#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_infer_request.hpp>
#include <ie_iinfer_request.hpp>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/ie_infer_request.hpp"

#define INVALID_ID -1

namespace py = pybind11;

class InferQueue {
public:
    InferQueue(std::vector<InferRequestWrapper> requests,
               std::queue<size_t> idle_handles,
               std::vector<py::object> user_ids)
        : _requests(requests),
          _idle_handles(idle_handles),
          _user_ids(user_ids) {
        this->setDefaultCallbacks();
        _last_id = -1;
    }

    ~InferQueue() {
        _requests.clear();
    }

    bool _is_ready() {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return !(_idle_handles.empty());
        });

        return !(_idle_handles.empty());
    }

    py::dict _getIdleRequestInfo() {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return !(_idle_handles.empty());
        });

        size_t request_id = _idle_handles.front();

        InferenceEngine::StatusCode status =
            _requests[request_id]._request.Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);

        if (status == InferenceEngine::StatusCode::RESULT_NOT_READY) {
            status = _requests[request_id]._request.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
        }

        py::dict request_info = py::dict();
        request_info["id"] = request_id;
        request_info["status"] = status;

        return request_info;
    }

    size_t getIdleRequestId() {
        // Wait for any of _idle_handles
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return !(_idle_handles.empty());
        });

        size_t idle_request_id = _idle_handles.front();
        _idle_handles.pop();

        return idle_request_id;
    }

    std::vector<InferenceEngine::StatusCode> waitAll() {
        // Wait for all requests to return with callback thus updating
        // _idle_handles so it matches the size of requests
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            return _idle_handles.size() == _requests.size();
        });

        std::vector<InferenceEngine::StatusCode> statuses;

        for (size_t handle = 0; handle < _requests.size(); handle++) {
            statuses.push_back(_requests[handle]._request.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY));
        }

        return statuses;
    }

    void setDefaultCallbacks() {
        for (size_t handle = 0; handle < _requests.size(); handle++) {
            _requests[handle]._request.SetCompletionCallback([this, handle /* ... */]() {
                _requests[handle]._endTime = Time::now();
                // Add idle handle to queue
                _idle_handles.push(handle);
                // Notify locks in getIdleRequestId() or waitAll() functions
                _cv.notify_one();
            });
        }
    }

    void setCustomCallbacks(py::function f_callback) {
        for (size_t handle = 0; handle < _requests.size(); handle++) {
            _requests[handle]._request.SetCompletionCallback([this, f_callback, handle /* ... */]() {
                _requests[handle]._endTime = Time::now();
                InferenceEngine::StatusCode statusCode =
                    _requests[handle]._request.Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
                if (statusCode == InferenceEngine::StatusCode::RESULT_NOT_READY) {
                    statusCode = InferenceEngine::StatusCode::OK;
                }
                // Acquire GIL, execute Python function
                py::gil_scoped_acquire acquire;
                f_callback(_requests[handle], statusCode, _user_ids[handle]);
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
    size_t _last_id;
    std::mutex _mutex;
    std::condition_variable _cv;
};

void regclass_InferQueue(py::module m) {
    py::class_<InferQueue, std::shared_ptr<InferQueue>> cls(m, "InferQueue");

    cls.def(py::init([](InferenceEngine::ExecutableNetwork& net, size_t jobs) {
                if (jobs == 0) {
                    const InferenceEngine::ExecutableNetwork& _net = net;
                    jobs = (size_t)Common::get_optimal_number_of_requests(_net);
                }

                std::vector<InferRequestWrapper> requests;
                std::queue<size_t> idle_handles;
                std::vector<py::object> user_ids(jobs);

                for (size_t handle = 0; handle < jobs; handle++) {
                    auto request = InferRequestWrapper(net.CreateInferRequest());
                    // Get Inputs and Outputs info from executable network
                    request._inputsInfo = net.GetInputsInfo();
                    request._outputsInfo = net.GetOutputsInfo();

                    requests.push_back(request);
                    idle_handles.push(handle);
                }

                return new InferQueue(requests, idle_handles, user_ids);
            }),
            py::arg("network"),
            py::arg("jobs") = 0);

    cls.def(
        "_async_infer",
        [](InferQueue& self, const py::dict inputs, py::object userdata) {
            // getIdleRequestId function has an intention to block InferQueue
            // until there is at least one idle (free to use) InferRequest
            auto handle = self.getIdleRequestId();
            // Set new inputs label/id from user
            self._user_ids[handle] = userdata;
            // Update inputs of picked InferRequest
            if (!inputs.empty()) {
                Common::set_request_blobs(self._requests[handle]._request, inputs);
            }
            // Now GIL can be released - we are NOT working with Python objects in this block
            {
                py::gil_scoped_release release;
                self._requests[handle]._startTime = Time::now();
                // Start InferRequest in asynchronus mode
                self._requests[handle]._request.StartAsync();
            }
        },
        py::arg("inputs"),
        py::arg("userdata"));

    cls.def("is_ready", [](InferQueue& self) {
        return self._is_ready();
    });

    cls.def("wait_all", [](InferQueue& self) {
        return self.waitAll();
    });

    cls.def("get_idle_request_info", [](InferQueue& self) {
        return self._getIdleRequestInfo();
    });

    cls.def("set_infer_callback", [](InferQueue& self, py::function f_callback) {
        self.setCustomCallbacks(f_callback);
    });

    cls.def("__len__", [](InferQueue& self) {
        return self._requests.size();
    });

    cls.def(
        "__iter__",
        [](InferQueue& self) {
            return py::make_iterator(self._requests.begin(), self._requests.end());
        },
        py::keep_alive<0, 1>()); /* Keep set alive while iterator is used */

    cls.def("__getitem__", [](InferQueue& self, size_t i) {
        return self._requests[i];
    });

    cls.def_property_readonly("userdata", [](InferQueue& self) {
        return self._user_ids;
    });
}
