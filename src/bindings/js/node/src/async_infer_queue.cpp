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

// TODO ? include pyopenvino/core/common.hpp get_optimal_number_of_requests
uint32_t get_optimal_number_of_requests(const ov::CompiledModel& actual) {
    try {
        auto supported_properties = actual.get_property(ov::supported_properties);
        OPENVINO_ASSERT(
            std::find(supported_properties.begin(), supported_properties.end(), ov::optimal_number_of_infer_requests) !=
                supported_properties.end(),
            "Can't load network: ",
            ov::optimal_number_of_infer_requests.name(),
            " is not supported!",
            " Please specify number of infer requests directly!");
        return actual.get_property(ov::optimal_number_of_infer_requests);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't load network: ", ex.what(), " Please specify number of infer requests directly!");
    }
}

AsyncInferQueue::AsyncInferQueue(const Napi::CallbackInfo& info) : Napi::ObjectWrap<AsyncInferQueue>(info) {
    auto env = info.Env();
    if (ov::js::validate<CompiledModelWrap, int>(info)) {
        const auto& cmodel_prototype = info.Env().GetInstanceData<AddonData>()->compiled_model;
        if (!cmodel_prototype ||
            !info[0].As<Napi::Object>().InstanceOf(cmodel_prototype.Value().As<Napi::Function>())) {
            reportError(info.Env(), "Invalid CompiledModel object.");
        }
        CompiledModelWrap* cm = Napi::ObjectWrap<CompiledModelWrap>::Unwrap(info[0].ToObject());
        auto& ocm = cm->get_compiled_model();
        size_t jobs = info[1].As<Napi::Number>().Int32Value();

        if (jobs == 0) {
            jobs = static_cast<size_t>(get_optimal_number_of_requests(ocm));
        }

        m_requests.reserve(jobs);
        m_user_ids.reserve(jobs);
        m_user_inputs.reserve(jobs);

        for (size_t handle = 0; handle < jobs; handle++) {
            // TODO ? Extend InferRequestWrapper to keep model inputs and outputs.
            m_requests.emplace_back(ocm.create_infer_request());
            m_user_ids.push_back(std::make_pair(Napi::Reference<Napi::Object>::New(Napi::Object::New(env), 2),
                                                Napi::Promise::Deferred::New(env)));
            m_user_inputs.push_back(Napi::Reference<Napi::Object>::New(Napi::Object::New(env), 2));
            m_idle_handles.push(handle);
        }
        this->set_default_callbacks();
    } else {
        reportError(info.Env(), "Invalid arguments. Expected CompiledModel and number of requests.");
    }
}

AsyncInferQueue::~AsyncInferQueue() {
    std::cout << "AsyncInferQueue destructor\n";
    m_requests.clear();
    this->tsfn.Release();  // release tsfn when all BlockingCalls are done
}

Napi::Function AsyncInferQueue::get_class(Napi::Env env) {
    return DefineClass(env,
                       "AsyncInferQueue",
                       {
                           InstanceMethod("getIdleRequestId", &AsyncInferQueue::get_idle_request_id),
                        //    InstanceMethod("waitAll", &AsyncInferQueue::wait_all),
                           InstanceMethod("setCallback", &AsyncInferQueue::set_custom_callbacks),
                           InstanceMethod("startAsync", &AsyncInferQueue::start_async),
                       });
}

Napi::Value AsyncInferQueue::get_idle_request_id(const Napi::CallbackInfo& info) {
    // TODO make it async method in js
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] {
        return !(m_idle_handles.empty());
    });
    size_t idle_handle = m_idle_handles.front();
    // wait for request to make sure it returned from callback
    m_requests[idle_handle].wait();
    if (m_errors.size() > 0)
        throw m_errors.front();
    return Napi::Number::New(info.Env(), idle_handle);
}

int AsyncInferQueue::check_idle_request_id() {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_idle_handles.empty()) {
        return -1;
    }
    int idle_handle = static_cast<int>(m_idle_handles.front());
    m_idle_handles.pop();
    // m_requests[idle_handle].wait(); // TODO request waits only for C++ callback to finish, not for js callback to
    // complete. Should work cause request is added to m_idle_handles after js callback is finished.
    if (m_errors.size() > 0)
        throw m_errors.front();  // TODO simulate such error
    return idle_handle;
}

void AsyncInferQueue::wait_all(const Napi::CallbackInfo& info) {
    // TODO does not work. Not needed now.
    for (auto& request : m_requests) {
        request.wait();
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_errors.size() > 0)
        throw m_errors.front();
}

void AsyncInferQueue::set_default_callbacks() {
    for (size_t handle = 0; handle < m_requests.size(); handle++) {
        m_requests[handle].set_callback([this, handle](std::exception_ptr exception_ptr) {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_idle_handles.push(handle);
            }
            m_cv.notify_one();
            try {
                if (exception_ptr) {
                    std::rethrow_exception(exception_ptr);
                }
            } catch (const std::exception& e) {
                OPENVINO_THROW(e.what());
            }
        });
    }
}

void AsyncInferQueue::set_custom_callbacks(const Napi::CallbackInfo& info) {
    // TODO add js::validate for function
    if (info[0].IsFunction()) {
        Napi::Function f_callback = info[0].As<Napi::Function>();
        this->tsfn =
            Napi::ThreadSafeFunction::New(info.Env(), f_callback, "AsyncInferQueueCallback", 0, 1, [](Napi::Env env) {
                std::cout << "Running tsfn finalizer callback\n";
            });
        for (size_t handle = 0; handle < m_requests.size(); handle++) {
            m_requests[handle].set_callback([this, handle](std::exception_ptr exception_ptr) {
                if (exception_ptr == nullptr) {
                    auto callback = [this](Napi::Env env, Napi::Function jsCallback, int* handle) {
                        // std::cout << "tsfn callback" << *handle << "\n";
                        Napi::Object js_ir = InferRequestWrap::wrap(env, m_requests[*handle]);
                        jsCallback.Call(
                            {js_ir, m_user_ids[*handle].first.Value()});  // performs whole callback and goes back here
                        auto promise = m_user_ids[*handle].second;
                        promise.Resolve(m_user_ids[*handle].first.Value());  // resolves promise, goes back here (does
                                                                             // not wait for promise.then() completion)
                        {
                            std::lock_guard<std::mutex> lock(m_mutex);
                            m_idle_handles.push(*handle);
                        }
                        m_cv.notify_one();  // TODO
                        {
                            // check if any requests are waiting
                            std::lock_guard<std::mutex> lock(m_mutex);
                            if (awaiting_requests.size() > 0) {
                                auto& request = awaiting_requests.front();
                                this->start_async_impl(*handle,
                                                       std::get<2>(request),
                                                       std::get<0>(request).Value(),
                                                       std::get<1>(request).Value());
                                awaiting_requests.pop();
                            }
                        }
                        delete handle;
                    };
                    // "callback" will be performed when main event loop is free
                    tsfn.BlockingCall(new int(handle), callback);
                }
                try {
                    if (exception_ptr) {
                        std::rethrow_exception(exception_ptr);
                    }
                } catch (const std::exception& e) {
                    OPENVINO_THROW(e.what());
                }
            });
        }
    } else {
        reportError(info.Env(), "Invalid callback. Expected function.");
    }
}

void AsyncInferQueue::start_async_impl(const int handle,
                                       Napi::Promise::Deferred deferred,
                                       Napi::Object infer_data,
                                       Napi::Object user_data) {
    m_user_ids[handle] = std::make_pair(Napi::Persistent(user_data), deferred);

    // TODO use parse_input_data(info[0].As<Napi::Value>());
    const auto& keys = infer_data.GetPropertyNames();
    for (uint32_t i = 0; i < keys.Length(); ++i) {
        auto input_name = static_cast<Napi::Value>(keys[i]).ToString().Utf8Value();
        auto value = infer_data.Get(input_name);
        m_user_inputs[handle] = Napi::Persistent(
            value.ToObject());  // keep reference to inputs so they are not garbage collected on js side
        auto tensor = value_to_tensor(value, m_requests[handle], input_name);

        m_requests[handle].set_tensor(input_name, tensor);
    }

    m_requests[handle].start_async();  // returns immediately, main event loop is free
}

Napi::Value AsyncInferQueue::start_async(const Napi::CallbackInfo& info) {
    // TODO validate Napi::CallbackInfo

    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(info.Env());

    const int handle = this->check_idle_request_id();
    if (handle == -1) {
        this->awaiting_requests.push(
            std::make_tuple(Napi::Persistent(info[0].ToObject()), Napi::Persistent(info[1].ToObject()), deferred));
    } else {
        this->start_async_impl(handle, deferred, info[0].ToObject(), info[1].ToObject());
    }
    return deferred.Promise();
}