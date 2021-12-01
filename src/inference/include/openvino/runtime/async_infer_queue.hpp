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
 * 
 * @note AsyncInferQueue provides simple and unifed interface for multiple asynchronous
 * calls to OpenVINO's InferRequests.
 */
class OPENVINO_RUNTIME_API AsyncInferQueue {
public:
    AsyncInferQueue() = delete;

    /**
     * @brief Basic constructor. Creates InferRequests based on given ExecutableNetwork.
     * 
     * @param net ExecutableNetwork object which will be base for creating of InferRequests.
     * @param jobs Number of InferRequests to be created. If equals to @p 0 AsyncInferQueue
     * will automatically try to choose number of jobs.
     */
    AsyncInferQueue(ExecutableNetwork &net, size_t jobs = 0);

    // ~AsyncInferQueue() {
    //     _requests.clear();
    // }

    /**
     * @brief Starts asynchronous inference on next avaiable job. Blocking call if there are no free jobs.
     * 
     * @param userdata Data passed to InferRequest's callback. Can be anything that matches ov::Any.
     */
    void start_async(const ov::Any userdata = nullptr);

    /**
     * @brief Starts asynchronous inference with given input data. Blocking call if there are no free jobs.
     *
     * @param inputs Map containing integers (index) and Tensor pairs.
     * @param userdata Data passed to InferRequest's callback. Can be anything that matches ov::Any.
     */
    void start_async(std::map<size_t, ov::runtime::Tensor> inputs, const ov::Any userdata = nullptr);

    /**
     * @brief Starts asynchronous inference with given input data. Blocking call if there are no free jobs.
     *
     * @param inputs Map containing strings (names) and Tensor pairs.
     * @param userdata Data passed to InferRequest's callback. Can be anything that matches ov::Any.
     */
    void start_async(std::map<std::string, ov::runtime::Tensor> inputs, const ov::Any userdata = nullptr);

    /**
     * @brief Waits for any request to become avaiable. Blocking call if there are no free jobs.
     *
     * @note Waits for InferRequest while queue is running.
     * @return Returns @p true if at least one request is avaiable, otherwise @p false.
     */
    bool is_ready();

    /**
     * @brief Waits for all jobs to be finished. This is a blocking call for caller thread until
     * all jobs are finished.
     */
    void wait_all();

    /**
     * @brief Gets id of the first idle job in queue.
     *
     * @return Returns integer.
     */
    size_t get_idle_id();

    /**
     * @brief Set callback function that will be called on success or failure for all jobs.
     *
     * @param callback callback object which will be called on when inference finish.
     * @note Callback function differs from single InferRequest's set_callback function. There is
     * a requirement to use InferRequest and Any as an argument. This allows queue's internals 
     * to pass underlaying request's object back to the user (for example: to get output tensors),
     * as well as providing additional data passing to the callback function.
     * 
     * It is NOT recommended to start another inference inside callback! (dodać o problemach itp)
     */
    void set_callback(std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback);

    /**
     * @brief Get InferRequest under given id from jobs pool.
     * 
     * @param i Specific job id.
     * 
     * @return Returns InferRequest from pool.
     */
    InferRequest& operator[](size_t i);

private:
    class Impl;
    std::unique_ptr<Impl, void (*)(Impl*)> m_pimpl; // = nullptr;
};

}  // namespace runtime
}  // namespace ov
