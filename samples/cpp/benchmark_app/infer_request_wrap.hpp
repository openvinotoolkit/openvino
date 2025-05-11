// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <openvino/openvino.hpp>
#include <queue>
#include <string>
#include <vector>

// clang-format off

#include "remote_tensors_filling.hpp"
#include "statistics_report.hpp"
#include "utils.hpp"
// clang-format on

typedef std::function<void(size_t id, size_t group_id, const double latency, const std::exception_ptr& ptr)>
    QueueCallbackFunction;

/// @brief Handles asynchronous callbacks and calculates execution time
class InferReqWrap final {
public:
    using Ptr = std::shared_ptr<InferReqWrap>;

    ~InferReqWrap() = default;

    explicit InferReqWrap(ov::CompiledModel& model, size_t id, QueueCallbackFunction callbackQueue)
        : _request(model.create_infer_request()),
          _id(id),
          _lat_group_id(0),
          _callbackQueue(callbackQueue),
          outputClBuffer() {
        _request.set_callback([&](const std::exception_ptr& ptr) {
            _endTime = Time::now();
            _callbackQueue(_id, _lat_group_id, get_execution_time_in_milliseconds(), ptr);
        });
    }

    void start_async() {
        _startTime = Time::now();
        _request.start_async();
    }

    void wait() {
        _request.wait();
    }

    void infer() {
        _startTime = Time::now();
        _request.infer();
        _endTime = Time::now();
        _callbackQueue(_id, _lat_group_id, get_execution_time_in_milliseconds(), nullptr);
    }

    std::vector<ov::ProfilingInfo> get_performance_counts() {
        return _request.get_profiling_info();
    }

    void set_shape(const std::string& name, const ov::Shape& dims) {
        // TODO check return status
        _request.get_tensor(name).set_shape(dims);
    }

    ov::Tensor get_tensor(const std::string& name) {
        return _request.get_tensor(name);
    }

    void set_tensor(const std::string& name, const ov::Tensor& data) {
        _request.set_tensor(name, data);
    }

    double get_execution_time_in_milliseconds() const {
        auto execTime = std::chrono::duration_cast<ns>(_endTime - _startTime);
        return static_cast<double>(execTime.count()) * 0.000001;
    }

    void set_latency_group_id(size_t id) {
        _lat_group_id = id;
    }

    // in case of using GPU memory we need to allocate CL buffer for
    // output blobs. By encapsulating cl buffer inside InferReqWrap
    // we will control the number of output buffers and access to it.
    std::map<std::string, ::gpu::BufferType>& get_output_cl_buffer() {
        return outputClBuffer;
    }

private:
    ov::InferRequest _request;
    Time::time_point _startTime;
    Time::time_point _endTime;
    size_t _id;
    size_t _lat_group_id;
    QueueCallbackFunction _callbackQueue;
    std::map<std::string, ::gpu::BufferType> outputClBuffer;
};

class InferRequestsQueue final {
public:
    InferRequestsQueue(ov::CompiledModel& model, size_t nireq, size_t lat_group_n, bool enable_lat_groups)
        : enable_lat_groups(enable_lat_groups) {
        for (size_t id = 0; id < nireq; id++) {
            requests.push_back(std::make_shared<InferReqWrap>(model,
                                                              id,
                                                              std::bind(&InferRequestsQueue::put_idle_request,
                                                                        this,
                                                                        std::placeholders::_1,
                                                                        std::placeholders::_2,
                                                                        std::placeholders::_3,
                                                                        std::placeholders::_4)));
            _idleIds.push(id);
        }
        _latency_groups.resize(lat_group_n);
        reset_times();
    }

    ~InferRequestsQueue() {
        // Inference Request guarantee that it will wait for all asynchronous internal tasks in destructor
        // So it should be released before any context that the request can use inside internal asynchronous tasks
        // For example all members of InferRequestsQueue would be destroyed before `requests` vector
        // So requests can try to use this members from `put_idle_request()` that would be called from request callback
        // To avoid this we should move this vector declaration after all members declaration or just clear it manually
        // in destructor
        requests.clear();
    }

    void reset_times() {
        _startTime = Time::time_point::max();
        _endTime = Time::time_point::min();
        _latencies.clear();
        for (auto& group : _latency_groups) {
            group.clear();
        }
    }

    double get_duration_in_milliseconds() {
        return std::chrono::duration_cast<ns>(_endTime - _startTime).count() * 0.000001;
    }

    void put_idle_request(size_t id,
                          size_t lat_group_id,
                          const double latency,
                          const std::exception_ptr& ptr = nullptr) {
        std::unique_lock<std::mutex> lock(_mutex);
        if (ptr) {
            inferenceException = ptr;
        } else {
            _latencies.push_back(latency);
            if (enable_lat_groups) {
                _latency_groups[lat_group_id].push_back(latency);
            }
            _idleIds.push(id);
            _endTime = std::max(Time::now(), _endTime);
        }
        _cv.notify_one();
    }

    InferReqWrap::Ptr get_idle_request() {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            if (inferenceException) {
                std::rethrow_exception(inferenceException);
            }
            return _idleIds.size() > 0;
        });
        auto request = requests.at(_idleIds.front());
        _idleIds.pop();
        _startTime = std::min(Time::now(), _startTime);
        return request;
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] {
            if (inferenceException) {
                std::rethrow_exception(inferenceException);
            }
            return _idleIds.size() == requests.size();
        });
    }

    std::vector<double> get_latencies() {
        return _latencies;
    }

    std::vector<std::vector<double>> get_latency_groups() {
        return _latency_groups;
    }

    std::vector<InferReqWrap::Ptr> requests;

private:
    std::queue<size_t> _idleIds;
    std::mutex _mutex;
    std::condition_variable _cv;
    Time::time_point _startTime;
    Time::time_point _endTime;
    std::vector<double> _latencies;
    std::vector<std::vector<double>> _latency_groups;
    bool enable_lat_groups;
    std::exception_ptr inferenceException = nullptr;
};
