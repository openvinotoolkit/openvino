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

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

class InferQueue
{
public:
    explicit InferQueue(InferenceEngine::ExecutableNetwork& net, size_t id, QueueCallbackFunction callbackQueue) :
        _request(net.CreateInferRequest()),
        _id(id),
        _callbackQueue(callbackQueue) {
        _request.SetCompletionCallback(
                [&]() {
                    _endTime = Time::now();
                    _callbackQueue(_id, getExecutionTimeInMilliseconds());
                });
    }

    void infer(size_t id, py::dict data);
    void start();
    void set_infer_callback();

private:
}

void regclass_InferQueue(py::module m);
