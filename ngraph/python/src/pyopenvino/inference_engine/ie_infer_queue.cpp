//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_infer_request.hpp>
#include <ie_common.h>
#include <ie_iinfer_request.hpp>

#include "pyopenvino/inference_engine/ie_infer_queue.hpp"

namespace py = pybind11;

class InferQueue
{
public:
    InferQueue(std::vector<InferenceEngine::InferRequest> requests, std::queue<size_t> idle_ids)
        : _requests(requests)
        , _idle_ids(idle_ids)
    {
    }

    ~InferQueue() { _requests.clear(); }

    size_t getIdleRequestId()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] { return !(_idle_ids.empty()); });

        size_t idle_request_id = _idle_ids.front();
        _idle_ids.pop();

        return idle_request_id;
    }

    std::vector<InferenceEngine::StatusCode> waitAll()
    {
        std::vector<InferenceEngine::StatusCode> statuses;

        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] { return _idle_ids.size() == _requests.size(); });

        for (size_t id = 0; id < _requests.size(); id++)
        {
            statuses.push_back(
                _requests[id].Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY));
        }

        return statuses;
    }

    void setCallbacks(py::function f_callback)
    {
        for (size_t id = 0; id < _requests.size(); id++)
        {
            _requests[id].SetCompletionCallback([this, f_callback, id /* ... */]() {
                py::gil_scoped_acquire acquire;
                f_callback(id /* request_id, result */);
                py::gil_scoped_release release;

                // lock queue and add idle id to queue
                // TODO: always set completion callback with this
                //       on creation of queue constructor (?)
                //       just in case no callback is given
                //       then for sure they want be push to queue
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _idle_ids.push(id);
                }
                _cv.notify_one();
            });
        }
    }

    std::vector<InferenceEngine::InferRequest> _requests;

private:
    std::queue<size_t> _idle_ids;
    std::mutex _mutex;
    std::condition_variable _cv;
};

void regclass_InferQueue(py::module m)
{
    py::class_<InferQueue, std::shared_ptr<InferQueue>> cls(m, "InferQueue");

    cls.def(py::init([](InferenceEngine::ExecutableNetwork& net, size_t jobs) {
        std::vector<InferenceEngine::InferRequest> requests;
        std::queue<size_t> idle_ids;
        for (size_t id = 0; id < jobs; id++)
        {
            requests.push_back(net.CreateInferRequest());
            idle_ids.push(id);
        }

        return new InferQueue(requests, idle_ids);
    }));

    cls.def("infer", [](InferQueue& self, const py::dict inputs) {
        py::gil_scoped_release release; // this makes func run async in python
        // getIdleRequestId function has an intention to block InferQueue (C++) until there is at least
        // one idle (free to use) InferRequest
        auto id = self.getIdleRequestId();
        // Update inputs of picked InferRequest
        for (auto&& input : inputs)
        {
            auto name = input.first.cast<std::string>().c_str();
            const std::shared_ptr<InferenceEngine::Blob>& blob =
                input.second.cast<const std::shared_ptr<InferenceEngine::Blob>&>();
            self._requests[id].SetBlob(name, blob);
        }
        // Start InferRequest in asynchronus mode
        self._requests[id].StartAsync();
    });

    cls.def("wait_all", [](InferQueue& self) {
        py::gil_scoped_release release;
        return self.waitAll();
    });

    cls.def("set_infer_callback",
            [](InferQueue& self, py::function f_callback) { self.setCallbacks(f_callback); });

    cls.def("__len__", [](InferQueue& self) { return self._requests.size(); });

    cls.def(
        "__iter__",
        [](InferQueue& self) {
            return py::make_iterator(self._requests.begin(), self._requests.end());
        },
        py::keep_alive<0, 1>()); /* Keep set alive while iterator is used */

    cls.def("__getitem__", [](InferQueue& self, size_t i) { return self._requests[i]; });
}
