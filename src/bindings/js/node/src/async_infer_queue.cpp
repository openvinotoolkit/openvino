// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/async_infer_queue.hpp"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "node/include/addon.hpp"
#include "node/include/compiled_model.hpp"
#include "node/include/errors.hpp"
#include "node/include/helper.hpp"
#include "node/include/infer_request.hpp"
#include "node/include/node_output.hpp"
#include "node/include/tensor.hpp"
#include "node/include/type_validation.hpp"

AsyncInferQueue::AsyncInferQueue(const Napi::CallbackInfo& info) : Napi::ObjectWrap<AsyncInferQueue>(info) {
    const auto env = info.Env();
    std::vector<std::string> allowed_signatures;

    try {
        const auto are_arguments_valid = ov::js::validate<CompiledModelWrap, int>(info, allowed_signatures);
        OPENVINO_ASSERT(are_arguments_valid,
                        "'AsyncInferQueue' constructor",
                        ov::js::get_parameters_error_msg(info, allowed_signatures));

        auto& compiled = Napi::ObjectWrap<CompiledModelWrap>::Unwrap(info[0].ToObject())->get_compiled_model();
        size_t jobs = info[1].As<Napi::Number>().Int32Value();
        m_tsfn = nullptr;

        m_requests.reserve(jobs);
        m_user_ids.reserve(jobs);
        m_user_inputs.reserve(jobs);

        for (size_t handle = 0; handle < jobs; handle++) {
            m_requests.emplace_back(compiled.create_infer_request());
            m_user_ids.push_back(std::make_pair(Napi::Reference<Napi::Object>::New(Napi::Object::New(env), 1),
                                                Napi::Promise::Deferred::New(env)));
            m_user_inputs.push_back(Napi::Reference<Napi::Object>::New(Napi::Object::New(env), 1));
            m_idle_handles.push(handle);
        }

    } catch (const ov::Exception& e) {
        reportError(env, e.what());
    }
}

AsyncInferQueue::~AsyncInferQueue() {
    m_requests.clear();
    m_user_ids.clear();
    m_user_inputs.clear();
}

void AsyncInferQueue::release(const Napi::CallbackInfo& info) {
    if (!m_tsfn) {
        return;
    }
    const auto status = m_tsfn.Release();
    if (status == napi_invalid_arg) {
        reportError(info.Env(), "Failed to release AsyncInferQueue thread-safe function. Its thread-count is zero.");
    } else if (status != napi_ok) {
        reportError(info.Env(), "Failed to release AsyncInferQueue thread-safe function.");
    }
}

Napi::Function AsyncInferQueue::get_class(Napi::Env env) {
    return DefineClass(env,
                       "AsyncInferQueue",
                       {
                           InstanceMethod("setCallback", &AsyncInferQueue::set_custom_callbacks),
                           InstanceMethod("startAsync", &AsyncInferQueue::start_async),
                           InstanceMethod("release", &AsyncInferQueue::release),
                       });
}

int AsyncInferQueue::check_idle_request_id() {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_idle_handles.empty()) {
        return -1;
    }
    auto idle_handle = static_cast<int>(m_idle_handles.front());
    // wait for request to make sure it returned from callback
    m_requests[idle_handle].wait();
    m_idle_handles.pop();
    return idle_handle;
}

void AsyncInferQueue::set_custom_callbacks(const Napi::CallbackInfo& info) {
    std::vector<std::string> allowed_signatures;
    try {
        OPENVINO_ASSERT(ov::js::validate<Napi::Function>(info, allowed_signatures),
                        "'set_callback'",
                        ov::js::get_parameters_error_msg(info, allowed_signatures));

        m_tsfn =
            Napi::ThreadSafeFunction::New(info.Env(), info[0].As<Napi::Function>(), "AsyncInferQueueCallback", 0, 1);
        for (size_t handle = 0; handle < m_requests.size(); handle++) {
            m_requests[handle].set_callback([this, handle](std::exception_ptr exception_ptr) {
                try {
                    if (exception_ptr) {
                        std::rethrow_exception(exception_ptr);
                    }
                    auto ov_callback = [this, handle](Napi::Env env, Napi::Function user_callback) {
                        Napi::Object js_ir = InferRequestWrap::wrap(env, m_requests[handle]);
                        const auto promise = m_user_ids[handle].second;
                        try {
                            user_callback.Call({js_ir, m_user_ids[handle].first.Value()});
                            promise.Resolve(m_user_ids[handle].first.Value());
                            // returns before the promise's .then() is completed
                        } catch (Napi::Error& e) {
                            promise.Reject(Napi::Error::New(env, e.Message()).Value());
                        }
                        {
                            // Start async inference on the next request or add idle handle to queue
                            std::lock_guard<std::mutex> lock(m_mutex);
                            if (m_awaiting_requests.size() > 0) {
                                auto& request = m_awaiting_requests.front();
                                start_async_impl(static_cast<int>(handle),
                                                 std::get<0>(request).Value(),
                                                 std::get<1>(request).Value(),
                                                 std::get<2>(request));
                                m_awaiting_requests.pop();
                            } else {
                                m_idle_handles.push(handle);
                            }
                        }
                    };
                    // The ov_callback will execute when the main event loop will be free
                    m_tsfn.BlockingCall(ov_callback);
                } catch (const std::exception& e) {
                    OPENVINO_THROW(e.what());
                }
            });
        }
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
    }
}

void AsyncInferQueue::start_async_impl(const int handle,
                                       Napi::Object infer_data,
                                       Napi::Object user_data,
                                       Napi::Promise::Deferred deferred) {
    m_user_inputs[handle] = Napi::Persistent(infer_data);  // keep reference to inputs so they are not garbage collected
    m_user_ids[handle] = std::make_pair(Napi::Persistent(user_data), deferred);

    // CVS-166764
    const auto& keys = infer_data.GetPropertyNames();
    for (uint32_t i = 0; i < keys.Length(); ++i) {
        auto input_name = static_cast<Napi::Value>(keys[i]).ToString().Utf8Value();
        auto value = infer_data.Get(input_name);
        auto tensor = value_to_tensor(value, m_requests[handle], input_name);

        m_requests[handle].set_tensor(input_name, tensor);
    }

    OPENVINO_ASSERT(m_tsfn != nullptr, "Callback has to be set before starting inference. Use 'setCallback' method.");
    m_requests[handle].start_async();  // returns immediately, main event loop is free
}

Napi::Value AsyncInferQueue::start_async(const Napi::CallbackInfo& info) {
    std::vector<std::string> allowed_signatures;

    try {
        const auto are_arguments_valid = ov::js::validate<Napi::Object, Napi::Value>(info, allowed_signatures);
        OPENVINO_ASSERT(are_arguments_valid,
                        "'startAsync'",
                        ov::js::get_parameters_error_msg(info, allowed_signatures));

        const auto handle = check_idle_request_id();
        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(info.Env());
        if (handle == -1) {
            m_awaiting_requests.push(
                std::make_tuple(Napi::Persistent(info[0].ToObject()), Napi::Persistent(info[1].ToObject()), deferred));
        } else {
            start_async_impl(handle, info[0].ToObject(), info[1].ToObject(), deferred);
        }
        return deferred.Promise();

    } catch (std::exception& err) {
        reportError(info.Env(), err.what());
        return info.Env().Undefined();
    }
}
