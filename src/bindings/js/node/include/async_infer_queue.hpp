// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "openvino/runtime/infer_request.hpp"

class AsyncInferQueue : public Napi::ObjectWrap<AsyncInferQueue> {
public:
    AsyncInferQueue(const Napi::CallbackInfo& info);
    static Napi::Function get_class(Napi::Env env);

    void release(const Napi::CallbackInfo& info);
    void set_custom_callbacks(const Napi::CallbackInfo& info);
    /**
     * @param info[0] Napi::Object containing data for inference.
     * @param info[1] Napi::Object containing user data that will be passed to the callback. [Optional]
     */
    Napi::Value start_async(const Napi::CallbackInfo& info);

private:
    int check_idle_request_id();
    void start_async_impl(const size_t handle,
                          Napi::Object infer_data,
                          Napi::Object user_data,
                          Napi::Promise::Deferred deferred);
    void set_tsfn(Napi::Env env, Napi::Function callback);
    void release();

    // AsyncInferQueue is the owner of all requests. When AsyncInferQueue is destroyed,
    // all of requests are destroyed as well.
    std::vector<ov::InferRequest> m_requests;
    std::vector<Napi::ObjectReference> m_user_inputs;  // to prevent garbage collection
    std::vector<std::pair<Napi::ObjectReference, Napi::Promise::Deferred>> m_user_ids;

    std::queue<size_t> m_idle_handles;
    std::queue<std::tuple<Napi::ObjectReference, Napi::ObjectReference, Napi::Promise::Deferred>> m_awaiting_requests;

    std::mutex m_mutex;
    Napi::ThreadSafeFunction m_tsfn;
};
