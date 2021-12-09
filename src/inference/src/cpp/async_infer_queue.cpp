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

namespace ov {
namespace runtime {
class AsyncInferQueue::Impl {
public:
    Impl() = default;

    Impl(ExecutableNetwork& net, size_t jobs) {
        // Automatically set number of jobs
        if (jobs == 0) {
            jobs = helpers::num_of_jobs_helper(net);
        }

        for (size_t handle = 0; handle < jobs; handle++) {
            m_pool.push_back(net.create_infer_request());
            m_idle_handles.push(handle);
            m_userdata.push_back(nullptr);
        }

        m_ref_pool = std::vector<std::reference_wrapper<ov::runtime::InferRequest>>(m_pool.begin(), m_pool.end());

        set_default_callback();
    }

    Impl(std::vector<std::reference_wrapper<ov::runtime::InferRequest>>&& ref_pool,
         std::queue<size_t>&& handles,
         std::vector<ov::Any>&& userdata)
        : m_ref_pool{std::move(ref_pool)},
          m_idle_handles{std::move(handles)},
          m_userdata{std::move(userdata)} {}

    InferRequest& get_request(size_t i);
    size_t get_pool_size();

    size_t get_idle_handle();
    size_t pop_idle_handle();

    void start_async(const ov::Any& userdata);
    void start_async(std::map<size_t, ov::runtime::Tensor>& inputs, const ov::Any& userdata);
    void start_async(std::map<std::string, ov::runtime::Tensor>& inputs, const ov::Any& userdata);

    bool is_ready();
    void wait_all();

    void set_callback(std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback);
    void set_job_callback(size_t handle,
                          std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback);

private:
    void set_default_callback();

    std::vector<ov::runtime::InferRequest> m_pool;                              // pool of jobs
    std::vector<std::reference_wrapper<ov::runtime::InferRequest>> m_ref_pool;  // pool of references to jobs
    std::queue<size_t> m_idle_handles;                                          // idle handles of requests
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<ov::Any> m_userdata;
};

InferRequest& AsyncInferQueue::Impl::get_request(size_t i) {
    return m_ref_pool[i];
}

size_t AsyncInferQueue::Impl::get_pool_size() {
    return m_ref_pool.size();
}

size_t AsyncInferQueue::Impl::get_idle_handle() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] {
        return !m_idle_handles.empty();
    });

    size_t handle = m_idle_handles.front();

    return handle;
}

size_t AsyncInferQueue::Impl::pop_idle_handle() {
    // Get idle request id and pop it from queue
    size_t idle_handle = get_idle_handle();
    std::unique_lock<std::mutex> lock(m_mutex);
    m_idle_handles.pop();

    return idle_handle;
}

void AsyncInferQueue::Impl::start_async(const ov::Any& userdata) {
    auto handle = pop_idle_handle();
    if (!userdata.empty()) {
        m_userdata[handle] = userdata;
    }
    m_ref_pool[handle].get().start_async();
}

void AsyncInferQueue::Impl::start_async(std::map<size_t, ov::runtime::Tensor>& inputs, const ov::Any& userdata) {
    size_t handle = pop_idle_handle();
    if (!userdata.empty()) {
        m_userdata[handle] = userdata;
    }
    for (auto const& t : inputs) {
        m_ref_pool[handle].get().set_input_tensor(t.first, t.second);
    }
    m_ref_pool[handle].get().start_async();
}

void AsyncInferQueue::Impl::start_async(std::map<std::string, ov::runtime::Tensor>& inputs, const ov::Any& userdata) {
    size_t handle = pop_idle_handle();
    if (!userdata.empty()) {
        m_userdata[handle] = userdata;
    }
    for (auto const& t : inputs) {
        m_ref_pool[handle].get().set_tensor(t.first, t.second);
    }
    m_ref_pool[handle].get().start_async();
}

bool AsyncInferQueue::Impl::is_ready() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] {
        return !m_idle_handles.empty();
    });

    return !(m_idle_handles.empty());
}

void AsyncInferQueue::Impl::wait_all() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] {
        return m_idle_handles.size() == m_ref_pool.size();
    });
}

void AsyncInferQueue::Impl::set_default_callback() {
    for (size_t handle = 0; handle < m_ref_pool.size(); handle++) {
        m_ref_pool[handle].get().set_callback([this, handle /* ... */](std::exception_ptr e) {
            // Add idle handle to queue
            std::unique_lock<std::mutex> lock(m_mutex);
            m_idle_handles.push(handle);
            // Notify locks
            m_cv.notify_one();
        });
    }
}

void AsyncInferQueue::Impl::set_callback(
    std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback) {
    for (size_t handle = 0; handle < m_ref_pool.size(); handle++) {
        set_job_callback(handle, callback);
    }
}

void AsyncInferQueue::Impl::set_job_callback(
    size_t handle,
    std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback) {
    m_ref_pool[handle].get().set_callback([this, callback, handle /* ... */](std::exception_ptr e) {
        callback(e, m_ref_pool[handle], m_userdata[handle]);
        // Add idle handle to queue
        std::unique_lock<std::mutex> lock(m_mutex);
        m_idle_handles.push(handle);
        // Notify locks
        m_cv.notify_one();
    });
}

AsyncInferQueue::AsyncInferQueue()
    : m_pimpl{new Impl{}, [](Impl* impl) {
                  delete impl;
              }} {}

AsyncInferQueue::AsyncInferQueue(ExecutableNetwork& net, size_t jobs)
    : m_pimpl{new Impl{net, jobs}, [](Impl* impl) {
                  delete impl;
              }} {}

AsyncInferQueue::AsyncInferQueue(std::vector<std::reference_wrapper<ov::runtime::InferRequest>>&& ref_pool,
                                 std::queue<size_t>&& handles,
                                 std::vector<ov::Any>&& userdata)
    : m_pimpl{new Impl{std::move(ref_pool), std::move(handles), std::move(userdata)}, [](Impl* impl) {
                  delete impl;
              }} {}

size_t AsyncInferQueue::get_idle_handle() {
    return m_pimpl->get_idle_handle();
}

void AsyncInferQueue::start_async(const ov::Any& userdata) {
    m_pimpl->start_async(userdata);
}

void AsyncInferQueue::start_async(std::map<size_t, ov::runtime::Tensor> inputs, const ov::Any& userdata) {
    m_pimpl->start_async(inputs, userdata);
}

void AsyncInferQueue::start_async(std::map<std::string, ov::runtime::Tensor> inputs, const ov::Any& userdata) {
    m_pimpl->start_async(inputs, userdata);
}

bool AsyncInferQueue::is_ready() {
    return m_pimpl->is_ready();
}

void AsyncInferQueue::wait_all() {
    m_pimpl->wait_all();
}

void AsyncInferQueue::set_callback(
    std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback) {
    m_pimpl->set_callback(callback);
}

void AsyncInferQueue::set_job_callback(
    size_t handle,
    std::function<void(std::exception_ptr, ov::runtime::InferRequest&, const ov::Any&)> callback) {
    m_pimpl->set_job_callback(handle, callback);
}

size_t AsyncInferQueue::size() {
    return m_pimpl->get_pool_size();
}

InferRequest& AsyncInferQueue::operator[](size_t i) {
    return m_pimpl->get_request(i);
}

}  // namespace runtime
}  // namespace ov
