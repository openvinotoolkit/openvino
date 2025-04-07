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

        for (size_t handle = 0; handle < jobs; handle++) {
            // TODO ? Extend InferRequestWrapper to keep model inputs and outputs.
            m_requests.emplace_back(ocm.create_infer_request());
            m_user_ids.push_back(Napi::Reference<Napi::Object>::New(Napi::Object::New(env), 2));
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
    this->tsfn.Release(); // release tsfn when all BlockingCalls are done
}

Napi::Function AsyncInferQueue::get_class(Napi::Env env) {
    return DefineClass(env,
                       "AsyncInferQueue",
                       {
                           InstanceMethod("getIdleRequestId", &AsyncInferQueue::get_idle_request_id),
                           InstanceMethod("waitAll", &AsyncInferQueue::wait_all),
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

void AsyncInferQueue::wait_all(const Napi::CallbackInfo& info) {
    // TODO make it async method in js
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
            });  // TODO when it should be deleted?
        for (size_t handle = 0; handle < m_requests.size(); handle++) {
            m_requests[handle].set_callback([this, handle](std::exception_ptr exception_ptr) {
                if (exception_ptr == nullptr) {
                    auto callback = [this](Napi::Env env, Napi::Function jsCallback, int* handle) {
                        // std::cout << "tsfn callback" << *handle << "\n";
                        Napi::Object js_ir = InferRequestWrap::wrap(env, m_requests[*handle]);
                        jsCallback.Call({js_ir, m_user_ids[*handle].Value()});
                        {
                            std::lock_guard<std::mutex> lock(m_mutex);
                            m_idle_handles.push(*handle);
                        }
                        m_cv.notify_one();
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

void AsyncInferQueue::start_async(const Napi::CallbackInfo& info) {
    // TODO validate Napi::CallbackInfo

    // getIdleRequestId waits for idle inferRequest and blocks main event loop
    auto handle = this->get_idle_request_id(info).As<Napi::Number>().Int32Value();
    m_user_ids[handle] = Napi::Persistent(info[1].ToObject());
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_idle_handles.pop();
    }

    // TODO use parse_input_data(info[0].As<Napi::Value>());
    const auto& keys = info[0].ToObject().GetPropertyNames();
    for (uint32_t i = 0; i < keys.Length(); ++i) {
        auto input_name = static_cast<Napi::Value>(keys[i]).ToString().Utf8Value();
        auto value = info[0].ToObject().Get(input_name);
        auto tensor = value_to_tensor(value, m_requests[handle], input_name);

        m_requests[handle].set_tensor(input_name, tensor);
    }

    m_requests[handle].start_async(); // returns immediately
    // main event loop is free
}