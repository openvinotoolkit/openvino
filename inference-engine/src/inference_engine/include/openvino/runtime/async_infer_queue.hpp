// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper classes for infer requests and callbacks.
 *
 * @file openvino/runtime/async_infer_queue.hpp
 */
#pragma once

#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/executable_network.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace runtime {

/**
 * @brief This is an interface of asynchronous infer request
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class OPENVINO_RUNTIME_API AsyncInferQueue {
public:
    /**
     * @brief Constructor
     */
    AsyncInferQueue(ov::runtime::ExecutableNetwork net, size_t jobs = 0);

    ~AsyncInferQueue() {
        _requests.clear();
    }

    /**
     * @brief Starts asynchronous inference on next avaiable job.
     */
    void start_async();

    /**
     * @brief Starts asynchronous inference with given input data.
     *
     * @param inputs Map containing integers (index) and Tensor pairs.
     */
    void start_async(std::map<size_t, ov::runtime::Tensor> inputs);

    /**
     * @brief Starts asynchronous inference with given input data.
     *
     * @param inputs Map containing strings (names) and Tensor pairs.
     */
    void start_async(std::map<std::string, ov::runtime::Tensor> inputs);

    /**
     * @brief Waits for any request to become avaiable.
     *
     * @note Waits for InferRequest while queue is running.
     * @return Returns @p true if at least one request is avaiable, otherwise @p false.
     */
    bool is_ready();

    /**
     * @brief Waits for all jobs to be finished.
     */
    void wait_all();

    /**
     * @brief Gets id of the first idle job in queue.
     *
     * @return Returns integer.
     */
    size_t get_idle_id();

    // Should this be a pointer?
    // ov::runtime::InferRequest* get_idle_request();
    // std::vector<ov::runtime::InferRequest*> get_requests();

    /**
     * @brief Set callback function that will be called on success or failure for all jobs.
     *
     * @param callback callback object which will be called on when inference finish.
     */
    void set_callback(std::function<void(std::exception_ptr, ov::runtime::InferRequest&)> callback);

protected:
    /**
     * @brief Gets id of the first idle job in queue and pops it of queue.
     *
     * @return Returns integer.
     */
    size_t pop_idle_id();

    /**
     * @brief Set callback function that controls flow for the queue.
     */
    void set_default_callback();

    std::vector<ov::runtime::InferRequest> _requests;
    std::queue<size_t> _idle_handles;
    std::mutex _mutex;
    std::condition_variable _cv;
    // std::vector<ov::Any> _user_ids;
};

}  // namespace runtime
}  // namespace ov
