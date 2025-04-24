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
    ~AsyncInferQueue();
    static Napi::Function get_class(Napi::Env env);

    void release(const Napi::CallbackInfo& info);
    void set_custom_callbacks(const Napi::CallbackInfo& info);
    Napi::Value start_async(const Napi::CallbackInfo& info);

private:
    int check_idle_request_id();
    void set_default_callbacks();
    void start_async_impl(const int handle,
                          Napi::Promise::Deferred deferred,
                          Napi::Object infer_data,
                          Napi::Object user_data);

    // AsyncInferQueue is the owner of all requests. When AsyncInferQueue is destroyed,
    // all of requests are destroyed as well.
    std::vector<ov::InferRequest> m_requests;
    std::vector<std::pair<Napi::ObjectReference, Napi::Promise::Deferred>> m_user_ids;
    std::vector<Napi::ObjectReference> m_user_inputs;  // to prevent garbage collection

    std::queue<size_t> m_idle_handles;
    std::queue<std::tuple<Napi::ObjectReference, Napi::ObjectReference, Napi::Promise::Deferred>> awaiting_requests;

    std::mutex m_mutex;
    std::queue<Napi::Error> m_errors;
    Napi::ThreadSafeFunction tsfn;
};
