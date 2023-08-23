// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "infer_request.hpp"

#include "compiled_model.hpp"
#include "node_output.hpp"
#include "tensor.hpp"

InferRequestWrap::InferRequestWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<InferRequestWrap>(info) {}

Napi::Function InferRequestWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "InferRequest",
                       {
                           InstanceMethod("setTensor", &InferRequestWrap::set_tensor),
                           InstanceMethod("setInputTensor", &InferRequestWrap::set_input_tensor),
                           InstanceMethod("setOuputTensor", &InferRequestWrap::set_output_tensor),
                           InstanceMethod("getTensor", &InferRequestWrap::get_tensor),
                           InstanceMethod("getInputTensor", &InferRequestWrap::get_input_tensor),
                           InstanceMethod("getOutputTensor", &InferRequestWrap::get_output_tensor),
                           InstanceMethod("getOutputTensors", &InferRequestWrap::get_output_tensors),
                           InstanceMethod("infer", &InferRequestWrap::infer_dispatch),
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

Napi::Value InferRequestWrap::set_tensor(const Napi::CallbackInfo& info) {
    if (info.Length() != 2 && !info[0].IsString()) {  // Add check info[1]
        reportError(info.Env(), "InferRequest.setTensor() invalid argument.");
    }
    std::string name = info[0].ToString();
    auto tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].ToObject());
    auto t = tensorWrap->get_tensor();

    _infer_request.set_tensor(name, t);
    return Napi::Value();
}

Napi::Value InferRequestWrap::set_input_tensor(const Napi::CallbackInfo& info) {
    if (info.Length() == 1) {
        auto tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].ToObject());
        auto t = tensorWrap->get_tensor();
        _infer_request.set_input_tensor(t);
    } else if (info.Length() == 2 && !info[0].IsNumber()) {  // Add check info[1]
        auto idx = info[0].ToNumber().Int32Value();
        auto tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].ToObject());
        auto t = tensorWrap->get_tensor();
        _infer_request.set_input_tensor(idx, t);
    } else {
        reportError(info.Env(), "InferRequest.setInputTensor() invalid argument.");
    }
    return Napi::Value();
}

Napi::Value InferRequestWrap::set_output_tensor(const Napi::CallbackInfo& info) {
    if (info.Length() == 1) {
        auto tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].ToObject());
        auto t = tensorWrap->get_tensor();
        _infer_request.set_output_tensor(t);
    } else if (info.Length() == 2 && !info[0].IsNumber()) {  // Add check info[1]
        auto idx = info[0].ToNumber().Int32Value();
        auto tensorWrap = Napi::ObjectWrap<TensorWrap>::Unwrap(info[0].ToObject());
        auto t = tensorWrap->get_tensor();
        _infer_request.set_output_tensor(idx, t);
    } else {
        reportError(info.Env(), "InferRequest.setOutputTensor() invalid argument.");
    }
    return Napi::Value();
}

Napi::Value InferRequestWrap::get_tensor(const Napi::CallbackInfo& info) {
    ov::Tensor tensor;
    if (info.Length() != 1) {
        reportError(info.Env(), "InferRequest.getTensor() invalid argument.");
    } else if (info[0].IsString()) {
        std::string tensor_name = info[0].ToString();
        tensor = _infer_request.get_tensor(tensor_name);
    } else {  // Add check info[0]
        auto outputWrap = Napi::ObjectWrap<Output<const ov::Node>>::Unwrap(info[0].ToObject());
        ov::Output<const ov::Node> output = outputWrap->get_output();
        tensor = _infer_request.get_tensor(output);
    }
    return TensorWrap::Wrap(info.Env(), tensor);
}

Napi::Value InferRequestWrap::get_input_tensor(const Napi::CallbackInfo& info) {
    ov::Tensor tensor;
    if (info.Length() == 0) {
        tensor = _infer_request.get_input_tensor();
    } else if (info.Length() == 1 && !info[0].IsNumber()) {  // Add check info[1]
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
    } else if (info.Length() == 1 && !info[0].IsNumber()) {  // Add check info[1]
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
    else if (info.Length() == 1 && info[0].IsArray()) {
        infer(info[0].As<Napi::Array>());
    } else if (info.Length() == 1 && info[0].IsObject()) {
        infer(info[0].As<Napi::Object>());
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
    auto compiled_model = _infer_request.get_compiled_model();
    return CompiledModelWrap::Wrap(info.Env(), compiled_model);
}
