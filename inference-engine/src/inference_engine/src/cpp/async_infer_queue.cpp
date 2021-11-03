// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/async_infer_queue.hpp"

#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "ie/ie_plugin_config.hpp"

namespace ov {
namespace runtime {

AsyncInferQueue::AsyncInferQueue(ov::runtime::ExecutableNetwork net, size_t jobs) {
    // Automatically set number of jobs
    if (jobs <= 0) {
        try {
            auto parameter_value = net.get_metric(METRIC_KEY(SUPPORTED_METRICS));
            auto supported_metrics = parameter_value.as<std::vector<std::string>>();
            const std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
            if (std::find(supported_metrics.begin(), supported_metrics.end(), key) != supported_metrics.end()) {
                parameter_value = net.get_metric(key);
                if (parameter_value.is<unsigned int>())
                    jobs = parameter_value.as<unsigned int>();
                else
                    IE_THROW() << "Unsupported format for " << key << "!"
                               << " Please specify number of infer requests directly!";
            } else {
                IE_THROW() << "Can't load network: " << key << " is not supported!"
                           << " Please specify number of infer requests directly!";
            }
        } catch (const std::exception& ex) {
            IE_THROW() << "Can't load network: " << ex.what() << " Please specify number of infer requests directly!";
        }
    }

    for (size_t handle = 0; handle < jobs; handle++) {
        _requests.push_back(net.create_infer_request());
        _idle_handles.push(handle);
    }

    set_default_callback();
}

size_t AsyncInferQueue::get_idle_id() {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] {
        return !(_idle_handles.empty());
    });

    size_t request_id = _idle_handles.front();

    return request_id;
}

size_t AsyncInferQueue::pop_idle_id() {
    // Get idle request id and pop it from queue
    size_t idle_request_id = get_idle_id();
    _idle_handles.pop();

    return idle_request_id;
}

void AsyncInferQueue::start_async() {
    _requests[AsyncInferQueue::pop_idle_id()].start_async();
}

void AsyncInferQueue::start_async(std::map<size_t, ov::runtime::Tensor> inputs) {
    size_t request_id = pop_idle_id();

    for (auto const& t : inputs) {
        _requests[request_id].set_input_tensor(t.first, t.second);
    }
    std::cout << std::endl << "Popped: " << request_id << std::endl;
    _requests[request_id].start_async();
}

void AsyncInferQueue::start_async(std::map<std::string, ov::runtime::Tensor> inputs) {
    size_t request_id = pop_idle_id();

    for (auto const& t : inputs) {
        _requests[request_id].set_tensor(t.first, t.second);
    }

    _requests[request_id].start_async();
}

bool AsyncInferQueue::is_ready() {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] {
        return !(_idle_handles.empty());
    });

    return !(_idle_handles.empty());
}

void AsyncInferQueue::wait_all() {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] {
        return _idle_handles.size() == _requests.size();
    });
}

// ov::runtime::InferRequest* get_idle_request();
// std::vector<ov::runtime::InferRequest*> get_requests();

void AsyncInferQueue::set_default_callback() {
    for (size_t handle = 0; handle < _requests.size(); handle++) {
        _requests[handle].set_callback([this, handle /* ... */](std::exception_ptr e) {
            // Add idle handle to queue
            _idle_handles.push(handle);
            // Notify locks
            _cv.notify_one();
        });
    }
}

void AsyncInferQueue::set_callback(std::function<void(std::exception_ptr, ov::runtime::InferRequest&)> callback) {
    for (size_t handle = 0; handle < _requests.size(); handle++) {
        _requests[handle].set_callback([this, callback, handle /* ... */](std::exception_ptr e) {
            callback(e, _requests[handle]);
            // Add idle handle to queue
            _idle_handles.push(handle);
            // Notify locks
            _cv.notify_one();
        });
    }
}

}  // namespace runtime
}  // namespace ov