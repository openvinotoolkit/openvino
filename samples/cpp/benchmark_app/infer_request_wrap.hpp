// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sys/stat.h>

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
#ifdef _WIN32
#    include "samples/os/windows/w_dirent.h"
#else
#    include <dirent.h>
#endif

// clang-format off

#include "remote_tensors_filling.hpp"
#include "statistics_report.hpp"
#include "utils.hpp"
#include "result_dump.hpp"
// clang-format on

typedef std::function<void(size_t id,
                           size_t group_id,
                           const double latency,
                           const InferenceResult& result,
                           const std::exception_ptr& ptr)>
    QueueCallbackFunction;

/// @brief Wrapper class for InferenceEngine::InferRequest. Handles asynchronous callbacks and calculates execution
/// time.
class InferReqWrap final {
public:
    using Ptr = std::shared_ptr<InferReqWrap>;

    ~InferReqWrap() = default;

    explicit InferReqWrap(ov::CompiledModel& model, size_t id, QueueCallbackFunction callbackQueue)
        : _request(model.create_infer_request()),
          _id(id),
          _lat_group_id(0),
          _callbackQueue(callbackQueue),
          _dump_output(false),
          outputClBuffer() {
        _request.set_callback([&](const std::exception_ptr& ptr) {
            _endTime = Time::now();
            InferenceResult inference_result;
            if (_dump_output) {
                const auto& compile_model = _request.get_compiled_model();
                const auto& outputs = compile_model.outputs();
                auto output_size = outputs.size();
                for (size_t i = 0; i < output_size; i++) {
                    const auto& tensor = _request.get_output_tensor(i);
                    inference_result.output_tensors.emplace_back(tensor);
                }
                inference_result.input_images = get_input_image_name();
            }
            _callbackQueue(_id, _lat_group_id, get_execution_time_in_milliseconds(), inference_result, ptr);
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
        InferenceResult result;
        if (_dump_output) {
            const auto& model = _request.get_compiled_model();
            const auto& outputs = model.outputs();
            auto output_size = outputs.size();
            for (size_t i = 0; i < output_size; i++) {
                result.output_tensors.emplace_back(_request.get_output_tensor(i));
            }
            result.input_images = get_input_image_name();
        }
        _callbackQueue(_id, _lat_group_id, get_execution_time_in_milliseconds(), result, nullptr);
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

    void set_input_image_name(const std::string& input_image_name) {
        _input_image_name = input_image_name;
    }

    std::string get_input_image_name() {
        return _input_image_name;
    }

    void set_dump_output(bool dump_output) {
        _dump_output = dump_output;
    }

private:
    ov::InferRequest _request;
    Time::time_point _startTime;
    Time::time_point _endTime;
    size_t _id;
    size_t _lat_group_id;
    QueueCallbackFunction _callbackQueue;
    bool _dump_output;
    std::map<std::string, ::gpu::BufferType> outputClBuffer;
    // If input size of model is greater than one, the input_images of inputs are joined with commas
    std::string _input_image_name;
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
                                                                        std::placeholders::_4,
                                                                        std::placeholders::_5)));
            _idleIds.push(id);
        }
        _latency_groups.resize(lat_group_n);
        reset_times();
    }

    ~InferRequestsQueue() {
        // Inference Request guarantee that it will wait for all asynchronous internal tasks in destructor
        // So it should be released before any context that the request can use inside internal asynchronous tasks
        // For example all members of InferRequestsQueue would be destroyed before `requests` vector
        // So requests can try to use this members from `putIdleRequest()` that would be called from request callback
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
                          const InferenceResult& result,
                          const std::exception_ptr& ptr = nullptr) {
        std::unique_lock<std::mutex> lock(_mutex);
        if (ptr) {
            inferenceException = ptr;
        } else {
            if (_result_dump) {
                _result_dump->compare_and_save_result(result.input_images, result.output_tensors);
            }
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
                try {
                    std::rethrow_exception(inferenceException);
                } catch (const std::exception& ex) {
                    throw ex;
                }
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
                try {
                    std::rethrow_exception(inferenceException);
                } catch (const std::exception& ex) {
                    throw ex;
                }
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

    void set_config(const std::string& device_name,
                    const ov::CompiledModel& model,
                    const std::string& dump_dir,
                    const uint32_t& output_max_num,
                    const uint32_t& binary_max_size) {
        struct stat dirstat;
        if (stat(dump_dir.c_str(), &dirstat) != 0) {
            slog::warn << dump_dir << " cannot be opened, dump_output is disabled" << slog::endl;
            return;
        }

        if (false == S_ISDIR(dirstat.st_mode)) {
            slog::warn << dump_dir << " is not a directory, dump_output is disabled" << slog::endl;
            return;
        }

        auto model_name = model.get_property(ov::model_name);
        auto output_size = model.outputs().size();
        std::string output_precision;
        for (size_t i = 0; i < output_size; i++) {
            auto type_name = model.output(i).get_element_type().get_type_name();
            output_precision += type_name + ",";
        }
        if (!output_precision.empty()) {
            output_precision.pop_back();
        }

        if (nullptr == _result_dump) {
            _result_dump = std::make_shared<ResultDump>(device_name,
                                                        model_name,
                                                        dump_dir,
                                                        output_precision,
                                                        output_max_num,
                                                        binary_max_size);
        }

        if (_result_dump) {
            for (auto& infer_req : requests) {
                infer_req->set_dump_output(true);
            }
        }
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
    std::shared_ptr<ResultDump> _result_dump;
};
