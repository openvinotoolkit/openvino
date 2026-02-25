// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/async_infer_queue.hpp"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace {
constexpr const char* UNDEFINED_USER_DATA = "UNDEFINED";
}

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
        const auto are_arguments_valid = ov::js::validate<CompiledModelWrap>(info, allowed_signatures) ||
                                         ov::js::validate<CompiledModelWrap, int>(info, allowed_signatures);
        OPENVINO_ASSERT(are_arguments_valid,
                        "'AsyncInferQueue' constructor",
                        ov::js::get_parameters_error_msg(info, allowed_signatures));

        auto& compiled = Napi::ObjectWrap<CompiledModelWrap>::Unwrap(info[0].ToObject())->get_compiled_model();
        size_t jobs =
            info.Length() == 1 ? get_optimal_number_of_requests(compiled) : info[1].As<Napi::Number>().Int32Value();
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

void AsyncInferQueue::release() {
    if (m_tsfn) {
        const auto status = m_tsfn.Release();
        OPENVINO_ASSERT(status == napi_ok, "Failed to release AsyncInferQueue resources.");
        m_tsfn = nullptr;
    }
}

void AsyncInferQueue::release(const Napi::CallbackInfo& info) {
    try {
        release();
    } catch (const ov::Exception& e) {
        reportError(info.Env(), e.what());
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
    m_idle_handles.pop();
    return idle_handle;
}

void AsyncInferQueue::set_tsfn(Napi::Env env, Napi::Function callback) {
    release();  // release previous ThreadSafeFunction if it exists
    m_tsfn = Napi::ThreadSafeFunction::New(env, callback, "AsyncInferQueueCallback", 0, 1);
}

void AsyncInferQueue::set_custom_callbacks(const Napi::CallbackInfo& info) {
    std::vector<std::string> allowed_signatures;
    try {
        OPENVINO_ASSERT(ov::js::validate<Napi::Function>(info, allowed_signatures),
                        "'setCallback'",
                        ov::js::get_parameters_error_msg(info, allowed_signatures));

        set_tsfn(info.Env(), info[0].As<Napi::Function>());
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
                            auto user_data =
                                m_user_ids[handle].first.Value().ToString().Utf8Value() == UNDEFINED_USER_DATA
                                    ? env.Undefined()
                                    : m_user_ids[handle].first.Value();
                            user_callback.Call({env.Null(), js_ir, user_data});  // CVS-170804
                            promise.Resolve(user_data);
                            // returns before the promise's .then() is completed
                        } catch (Napi::Error& e) {
                            promise.Reject(Napi::Error::New(env, e.Message()).Value());
                        }
                        // Start async inference on the next request or add idle handle to queue
                        if (std::lock_guard<std::mutex> lock(m_mutex); m_awaiting_requests.size() > 0) {
                            const auto& [infer_data, user_data, promise] = m_awaiting_requests.front();
                            start_async_impl(handle, infer_data.Value(), user_data.Value(), promise);
                            m_awaiting_requests.pop();
                        } else {
                            m_idle_handles.push(handle);
                        }
                    };
                    // The ov_callback will execute when the main event loop will be free
                    napi_status status = m_tsfn.BlockingCall(ov_callback);
                    OPENVINO_ASSERT(status == napi_ok, "Failed to call user callback.");
                } catch (const std::exception& e) {
                    OPENVINO_THROW(e.what());
                }
            });
        }
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
    }
}

void AsyncInferQueue::start_async_impl(const size_t handle,
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
        const auto are_arguments_valid = ov::js::validate<Napi::Object>(info, allowed_signatures) ||
                                         ov::js::validate<Napi::Object, Napi::Value>(info, allowed_signatures);
        OPENVINO_ASSERT(are_arguments_valid,
                        "'startAsync'",
                        ov::js::get_parameters_error_msg(info, allowed_signatures));

        const auto handle = check_idle_request_id();
        // WA for "Error: Invalid argument" when Napi::Object is undefined.
        auto user_data =
            info.Length() > 1 ? info[1].ToObject() : Napi::String::New(info.Env(), UNDEFINED_USER_DATA).ToObject();
        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(info.Env());
        if (handle == -1) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_awaiting_requests.push(
                std::make_tuple(Napi::Persistent(info[0].ToObject()), Napi::Persistent(user_data), deferred));
        } else {
            start_async_impl(handle, info[0].ToObject(), user_data, deferred);
        }
        return deferred.Promise();

    } catch (std::exception& err) {
        reportError(info.Env(), err.what());
        return info.Env().Undefined();
    }
}
