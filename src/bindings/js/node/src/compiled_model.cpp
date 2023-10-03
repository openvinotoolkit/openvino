// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "compiled_model.hpp"

#include "errors.hpp"
#include "infer_request.hpp"
#include "node_output.hpp"

CompiledModelWrap::CompiledModelWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<CompiledModelWrap>(info) {}

Napi::Function CompiledModelWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "CompiledModel",
                       {InstanceMethod("createInferRequest", &CompiledModelWrap::create_infer_request),
                        InstanceMethod("input", &CompiledModelWrap::get_input),
                        InstanceAccessor<&CompiledModelWrap::get_inputs>("inputs"),
                        InstanceMethod("output", &CompiledModelWrap::get_output),
                        InstanceAccessor<&CompiledModelWrap::get_outputs>("outputs")});
}

Napi::Object CompiledModelWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("CompiledModel", func);
    return exports;
}

Napi::Object CompiledModelWrap::Wrap(Napi::Env env, ov::CompiledModel compiled_model) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    CompiledModelWrap* cm = Napi::ObjectWrap<CompiledModelWrap>::Unwrap(obj);
    cm->_compiled_model = compiled_model;
    return obj;
}

void CompiledModelWrap::set_compiled_model(const ov::CompiledModel& compiled_model) {
    _compiled_model = compiled_model;
}

Napi::Value CompiledModelWrap::create_infer_request(const Napi::CallbackInfo& info) {
    ov::InferRequest infer_request = _compiled_model.create_infer_request();
    return InferRequestWrap::Wrap(info.Env(), infer_request);
}

Napi::Value CompiledModelWrap::get_output(const Napi::CallbackInfo& info) {
    if (info.Length() == 0) {
        try {
            return Output<const ov::Node>::Wrap(info.Env(), _compiled_model.output());
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
            return Napi::Value();
        }
    } else if (info.Length() != 1) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else if (info[0].IsString()) {
        auto tensor_name = info[0].ToString();
        return Output<const ov::Node>::Wrap(info.Env(), _compiled_model.output(tensor_name));
    } else if (info[0].IsNumber()) {
        auto idx = info[0].As<Napi::Number>().Int32Value();
        return Output<const ov::Node>::Wrap(info.Env(), _compiled_model.output(idx));
    } else {
        reportError(info.Env(), "Error while getting compiled model outputs.");
        return Napi::Value();
    }
}

Napi::Value CompiledModelWrap::get_outputs(const Napi::CallbackInfo& info) {
    auto cm_outputs = _compiled_model.outputs();  // Output<Node>
    Napi::Array js_outputs = Napi::Array::New(info.Env(), cm_outputs.size());

    size_t i = 0;
    for (auto& out : cm_outputs)
        js_outputs[i++] = Output<const ov::Node>::Wrap(info.Env(), out);

    return js_outputs;
}

Napi::Value CompiledModelWrap::get_input(const Napi::CallbackInfo& info) {
    if (info.Length() == 0) {
        try {
            return Output<const ov::Node>::Wrap(info.Env(), _compiled_model.input());
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
            return Napi::Value();
        }
    } else if (info.Length() != 1) {
        reportError(info.Env(), "Invalid number of arguments -> " + std::to_string(info.Length()));
        return Napi::Value();
    } else if (info[0].IsString()) {
        auto tensor_name = info[0].ToString();
        return Output<const ov::Node>::Wrap(info.Env(), _compiled_model.input(tensor_name));
    } else if (info[0].IsNumber()) {
        auto idx = info[0].As<Napi::Number>().Int32Value();
        return Output<const ov::Node>::Wrap(info.Env(), _compiled_model.input(idx));
    } else {
        reportError(info.Env(), "Error while getting compiled model inputs.");
        return Napi::Value();
    }
}

Napi::Value CompiledModelWrap::get_inputs(const Napi::CallbackInfo& info) {
    auto cm_inputs = _compiled_model.inputs();  // Output<Node>
    Napi::Array js_inputs = Napi::Array::New(info.Env(), cm_inputs.size());

    size_t i = 0;
    for (auto& out : cm_inputs)
        js_inputs[i++] = Output<const ov::Node>::Wrap(info.Env(), out);

    return js_inputs;
}
