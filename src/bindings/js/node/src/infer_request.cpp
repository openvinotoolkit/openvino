// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "infer_request.hpp"

#include <mutex>
#include <random>
#include <thread>

#include "compiled_model.hpp"
#include "node_output.hpp"
#include "tensor.hpp"

std::mutex infer_mutex;

InferRequestWrap::InferRequestWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<InferRequestWrap>(info) {}

Napi::Function InferRequestWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "InferRequest",
                       {
                           InstanceMethod("setTensor", &InferRequestWrap::set_tensor),
                           InstanceMethod("setInputTensor", &InferRequestWrap::set_input_tensor),
                           InstanceMethod("setOutputTensor", &InferRequestWrap::set_output_tensor),
                           InstanceMethod("getTensor", &InferRequestWrap::get_tensor),
                           InstanceMethod("getInputTensor", &InferRequestWrap::get_input_tensor),
                           InstanceMethod("getOutputTensor", &InferRequestWrap::get_output_tensor),
                           InstanceMethod("infer", &InferRequestWrap::infer_dispatch),
                           InstanceMethod("inferAsync", &InferRequestWrap::infer_async),
                           InstanceMethod("getCompiledModel", &InferRequestWrap::get_compiled_model),
                       });
}

Napi::Object InferRequestWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("InferRequest", func);
    return exports;
}

void InferRequestWrap::set_infer_request(const ov::InferRequest& infer_request) {
    _infer_request = infer_request;
}

Napi::Object InferRequestWrap::Wrap(Napi::Env env, ov::InferRequest infer_request) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    InferRequestWrap* ir = Napi::ObjectWrap<InferRequestWrap>::Unwrap(obj);
    ir->set_infer_request(infer_request);
    return obj;
}

void InferRequestWrap::set_tensor(const Napi::CallbackInfo& info) {
    if (info.Length() != 2 || !info[0].IsString() || !info[1].IsObject()) {
        reportError(info.Env(), "InferRequest.setTensor() invalid argument.");
    } else {
        std::string name = info[0].ToString();
        auto tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].ToObject());
        _infer_request.set_tensor(name, tensorWrap->get_tensor());
    }
}

void InferRequestWrap::set_input_tensor(const Napi::CallbackInfo& info) {
    if (info.Length() == 1 && info[0].IsObject()) {
        auto tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].ToObject());
        _infer_request.set_input_tensor(tensorWrap->get_tensor());
    } else if (info.Length() == 2 && info[0].IsNumber() && info[1].IsObject()) {
        auto idx = info[0].ToNumber().Int32Value();
        auto tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].ToObject());
        _infer_request.set_input_tensor(idx, tensorWrap->get_tensor());
    } else {
        reportError(info.Env(), "InferRequest.setInputTensor() invalid argument.");
    }
}

void InferRequestWrap::set_output_tensor(const Napi::CallbackInfo& info) {
    if (info.Length() == 1) {
        auto tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].ToObject());
        auto t = tensorWrap->get_tensor();
        _infer_request.set_output_tensor(t);
    } else if (info.Length() == 2 && info[0].IsNumber() && info[1].IsObject()) {
        auto idx = info[0].ToNumber().Int32Value();
        auto tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[1].ToObject());
        _infer_request.set_output_tensor(idx, tensorWrap->get_tensor());
    } else {
        reportError(info.Env(), "InferRequest.setOutputTensor() invalid argument.");
    }
}

Napi::Value InferRequestWrap::get_tensor(const Napi::CallbackInfo& info) {
    ov::Tensor tensor;
    if (info.Length() != 1) {
        reportError(info.Env(), "InferRequest.getTensor() invalid number of arguments.");
    } else if (info[0].IsString()) {
        std::string tensor_name = info[0].ToString();
        tensor = _infer_request.get_tensor(tensor_name);
    } else if (info[0].IsObject()) {
        auto outputWrap = Napi::ObjectWrap<Output<const ov::Node>>::Unwrap(info[0].ToObject());
        ov::Output<const ov::Node> output = outputWrap->get_output();
        tensor = _infer_request.get_tensor(output);
    } else {
        reportError(info.Env(), "InferRequest.getTensor() invalid argument.");
    }
    return TensorWrap::Wrap(info.Env(), tensor);
}

Napi::Value InferRequestWrap::get_input_tensor(const Napi::CallbackInfo& info) {
    ov::Tensor tensor;
    if (info.Length() == 0) {
        tensor = _infer_request.get_input_tensor();
    } else if (info.Length() == 1 && info[0].IsNumber()) {
        auto idx = info[0].ToNumber().Int32Value();
        tensor = _infer_request.get_input_tensor(idx);
    } else {
        reportError(info.Env(), "InferRequest.getInputTensor() invalid argument.");
    }
    return TensorWrap::Wrap(info.Env(), tensor);
}

Napi::Value InferRequestWrap::get_output_tensor(const Napi::CallbackInfo& info) {
    ov::Tensor tensor;
    if (info.Length() == 0) {
        tensor = _infer_request.get_output_tensor();
    } else if (info.Length() == 1 && info[0].IsNumber()) {
        auto idx = info[0].ToNumber().Int32Value();
        tensor = _infer_request.get_output_tensor(idx);
    } else {
        reportError(info.Env(), "InferRequest.getInputTensor() invalid argument.");
    }
    return TensorWrap::Wrap(info.Env(), tensor);
}

Napi::Value InferRequestWrap::get_output_tensors(const Napi::CallbackInfo& info) {
    auto compiled_model = _infer_request.get_compiled_model().outputs();
    auto outputs_obj = Napi::Object::New(info.Env());

    for (auto& node : compiled_model) {
        auto tensor = _infer_request.get_tensor(node);
        auto new_tensor = ov::Tensor(tensor.get_element_type(), tensor.get_shape());
        tensor.copy_to(new_tensor);
        outputs_obj.Set(node.get_any_name(), TensorWrap::Wrap(info.Env(), new_tensor));
    }
    return outputs_obj;
}

Napi::Value InferRequestWrap::infer_dispatch(const Napi::CallbackInfo& info) {
    if (info.Length() == 0)
        _infer_request.infer();
    else if (info.Length() == 1 && info[0].IsTypedArray()) {
        reportError(info.Env(), "TypedArray cannot be passed directly into infer() method.");
        return info.Env().Null();
    } else if (info.Length() == 1 && info[0].IsArray()) {
        try {
            infer(info[0].As<Napi::Array>());
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
            return info.Env().Null();
        }
    } else if (info.Length() == 1 && info[0].IsObject()) {
        try {
            infer(info[0].As<Napi::Object>());
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
            return info.Env().Null();
        }
    } else {
        reportError(info.Env(), "Infer method takes as an argument an array or an object.");
    }
    return get_output_tensors(info);
}

void InferRequestWrap::infer(const Napi::Array& inputs) {
    for (size_t i = 0; i < inputs.Length(); ++i) {
        auto tensor = value_to_tensor(inputs[i], _infer_request, i);
        _infer_request.set_input_tensor(i, tensor);
    }
    _infer_request.infer();
}

void InferRequestWrap::infer(const Napi::Object& inputs) {
    auto keys = inputs.GetPropertyNames();

    for (size_t i = 0; i < keys.Length(); ++i) {
        auto input_name = static_cast<Napi::Value>(keys[i]).ToString().Utf8Value();
        auto value = inputs.Get(input_name);
        auto tensor = value_to_tensor(value, _infer_request, input_name);

        _infer_request.set_tensor(input_name, tensor);
    }
    _infer_request.infer();
}

Napi::Value InferRequestWrap::get_compiled_model(const Napi::CallbackInfo& info) {
    return CompiledModelWrap::Wrap(info.Env(), _infer_request.get_compiled_model());
}

struct TsfnContext {
    TsfnContext(Napi::Env env) : deferred(Napi::Promise::Deferred::New(env)){};

    Napi::Promise::Deferred deferred;
    ov::InferRequest* _context_ir;
    std::thread nativeThread;

    Napi::ThreadSafeFunction tsfn;
    std::vector<ov::Tensor> input_tensors;
    std::map<std::string, ov::Tensor> result;
};

void FinalizerCallback(Napi::Env env, void* finalizeData, TsfnContext* context) {
    context->nativeThread.join();
    delete context;
};

void performInferenceThread(TsfnContext* data) {
    infer_mutex.lock();
    for (size_t i = 0; i < data->input_tensors.size(); ++i) {
        data->_context_ir->set_input_tensor(i, data->input_tensors[i]);
    }
    data->_context_ir->infer();

    auto compiled_model = data->_context_ir->get_compiled_model().outputs();
    std::map<std::string, ov::Tensor> outputs;

    for (auto& node : compiled_model) {
        auto tensor = data->_context_ir->get_tensor(node);
        auto new_tensor = ov::Tensor(tensor.get_element_type(), tensor.get_shape());
        tensor.copy_to(new_tensor);
        outputs.insert({node.get_any_name(), new_tensor});
    }

    data->result = outputs;
    infer_mutex.unlock();

    auto callback = [](Napi::Env env, Napi::Function, TsfnContext* data) {
        auto m = data->result;
        auto outputs_obj = Napi::Object::New(env);

        for (const auto& [key, tensor] : m) {
            outputs_obj.Set(key, TensorWrap::Wrap(env, tensor));
        }
        data->deferred.Resolve({outputs_obj});
    };

    data->tsfn.BlockingCall(data, callback);
    data->tsfn.Release();
}


Napi::Value InferRequestWrap::infer_async(const Napi::CallbackInfo& info) {
    if (info.Length() != 1) {
        reportError(info.Env(), "InferAsync method takes as an argument an array or an object.");
    }
    Napi::Env env = info.Env();

    auto context_data = new TsfnContext(env);
    context_data->_context_ir = &_infer_request;
    try {
        auto parsed_input = parse_input_data(info[0]);
        context_data->input_tensors = parsed_input;
    } catch (std::exception& e) {
        reportError(info.Env(), e.what());
    }

    context_data->tsfn = Napi::ThreadSafeFunction::New(env,
                                                       Napi::Function(),
                                                       "TSFN",
                                                       0,
                                                       1,
                                                       context_data,
                                                       FinalizerCallback,
                                                       (void*)nullptr);

    context_data->nativeThread = std::thread(performInferenceThread, context_data);
    return context_data->deferred.Promise();
}
