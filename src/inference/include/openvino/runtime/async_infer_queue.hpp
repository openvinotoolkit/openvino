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

#include "ie/ie_plugin_config.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/executable_network.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace helpers {
/**
 * @brief Helper function to get number of jobs based on given network's metrics.
 *
 * @param net ExecutableNetwork object.
 *
 * @return Returns number of jobs.
 */
static size_t num_of_jobs_helper(ov::runtime::ExecutableNetwork& net) {
    try {
        auto parameter_value = net.get_metric(METRIC_KEY(SUPPORTED_METRICS));
        auto supported_metrics = parameter_value.as<std::vector<std::string>>();
        const std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
        if (std::find(supported_metrics.begin(), supported_metrics.end(), key) != supported_metrics.end()) {
            parameter_value = net.get_metric(key);
            if (parameter_value.is<unsigned int>())
                return parameter_value.as<unsigned int>();
            else
                throw ov::Exception("Unsupported format for " + key + "! Please specify number of jobs explicitly!");
        } else {
            throw ov::Exception("Can't load model: " + key +
                                " is not supported! Please specify number of jobs explicitly!");
        }
    } catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    }
}
}  // namespace helpers

namespace ov {
namespace runtime {

/**
 * @brief This is an interface of asynchronous inference of requests.
 *
 * Based on a concept of jobs pool which holds InferRequests.
 * Work is automatically dispatched without user synchronization inside one queue.
 * Advanced workflow can be managed via callbacks and blocking functions like
 * wait_all() or is_ready().
 *
 * @note AsyncInferQueue provides simple and unified interface for multiple asynchronous
 * calls to OpenVINO's InferRequests.
 */
class OPENVINO_RUNTIME_API AsyncInferQueue {
public:
    /**
     * @brief Empty constructor.
     */
    AsyncInferQueue();

    /**
     * @brief Basic constructor. Creates InferRequests based on given ExecutableNetwork.
     *
     * @param net ExecutableNetwork object which will be base for creating of InferRequests.
     * @param jobs number of InferRequests to be created. If equals to @p 0 AsyncInferQueue
     * will automatically try to choose number of jobs.
     */
    AsyncInferQueue(ExecutableNetwork& net, size_t jobs = 0);

    /**
     * @brief Advanced constructor. Creates AsyncInferQueue based on given vectors.
     *
     * @note It allows to use queue as a part of a compostition pattern.
     * Example can be seen in Python based bindings of AsyncInferQueue.
     *
     * @param ref_pool vector of std references to already created InferReuqests.
     * @param handles queue with handles, should be from 0 to size of the pool.
     * @param userdata vector with userdata - ov::Any objects.
     */
    AsyncInferQueue(std::vector<std::reference_wrapper<ov::runtime::InferRequest>>&& ref_pool,
                    std::queue<size_t>&& handles,
                    std::vector<ov::Any>&& userdata);

    /**
     * @brief Starts asynchronous inference on next avaiable job. Blocking call if there are no free jobs.
     *
     * @param userdata data passed to InferRequest's callback. Can be anything that matches ov::Any.
     */
    void start_async(const ov::Any userdata = nullptr);

    /**
     * @brief Starts asynchronous inference with given input data. Blocking call if there are no free jobs.
     *
     * @param inputs map containing integers (index) and Tensor pairs.
     * @param userdata data passed to InferRequest's callback. Can be anything that matches ov::Any.
     */
    void start_async(std::map<size_t, ov::runtime::Tensor> inputs, const ov::Any userdata = nullptr);

    /**
     * @brief Starts asynchronous inference with given input data. Blocking call if there are no free jobs.
     *
     * @param inputs map containing strings (names) and Tensor pairs.
     * @param userdata data passed to InferRequest's callback. Can be anything that matches ov::Any.
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
     *
     * @note Call should be at the end of every AsyncInferQueue's "work scope". After wait_all()
     * user is free to interact with all jobs (InferRequest objects) from the pool.
     */
    void wait_all();

    /**
     * @brief Gets handle of the first idle job in queue.
     *
     * @return Returns integer.
     */
    size_t get_idle_handle();

    /**
     * @brief Set callback function that will be called on success or failure for all jobs.
     *
     * @param callback callback function which will be called when inference finish.
     * @note Callback function differs from single InferRequest's set_callback function. There is
     * a requirement to use InferRequest and Any as an argument. This allows queue's internals
     * to pass underlaying request's object back to the user (for example: to get output tensors),
     * as well as providing additional data passing to the callback function.
     *
     * It is NOT recommended to start another inference inside callback!
     * This can lead to worse performance or break execution if not synchronized correctly.
     */
    void set_callback(std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback);

    /**
     * @brief Set callback function that will be called on success or failure for specified job.
     *
     * @param handle handle to job, must be less then size of the queue.
     * @param callback callback function which will be called when inference finish.
     * @note Callback function differs from single InferRequest's set_callback function. There is
     * a requirement to use InferRequest and Any as an argument. This allows queue's internals
     * to pass underlaying request's object back to the user (for example: to get output tensors),
     * as well as providing additional data passing to the callback function.
     *
     * It is NOT recommended to start another inference inside callback!
     * This can lead to worse performance or break execution if not synchronized correctly.
     */
    void set_job_callback(size_t handle,
                          std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback);

    /**
     * @brief Get size of the pool.
     *
     * @return Returns size_t equal to number of jobs in the pool.
     */
    size_t size();

    /**
     * @brief Get InferRequest under given handle from jobs pool.
     *
     * @param i specific job handle.
     *
     * @return Returns InferRequest from pool.
     */
    InferRequest& operator[](size_t i);

private:
    class Impl;
    std::unique_ptr<Impl, void (*)(Impl*)> m_pimpl;
};

}  // namespace runtime
}  // namespace ov
