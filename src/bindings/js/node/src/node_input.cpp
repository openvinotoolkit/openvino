// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node_input.hpp"
#include "helper.hpp"

Input<ov::Node>::Input(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Input<ov::Node>>(info) {}

Napi::Function Input<ov::Node>::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "Input",
                       {Input<ov::Node>::InstanceMethod("getShape", &Input<ov::Node>::get_shape),
                        Input<ov::Node>::InstanceAccessor<&Input<ov::Node>::get_shape>("shape")});
}

Napi::Object Input<ov::Node>::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Input", func);
    return exports;
}

void Input<ov::Node>::set_input(const std::shared_ptr<ov::Input<ov::Node>>& input) {
    _input = input;
}

Napi::Object Input<ov::Node>::Wrap(Napi::Env env, std::shared_ptr<ov::Input<ov::Node>> input) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    Input* input_ptr = Napi::ObjectWrap<Input>::Unwrap(obj);
    input_ptr->set_input(input);
    return obj;
}

Napi::Value Input<ov::Node>::get_shape(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::Shape, Napi::Array>(info, _input->get_shape());
}

Input<const ov::Node>::Input(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Input<const ov::Node>>(info) {}

Napi::Function Input<const ov::Node>::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "Input",
                       {Input<const ov::Node>::InstanceMethod("getShape", &Input<const ov::Node>::get_shape),
                        Input<const ov::Node>::InstanceAccessor<&Input<const ov::Node>::get_shape_data>("shape")});
}

Napi::Object Input<const ov::Node>::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Input", func);
    return exports;
}

void Input<const ov::Node>::set_input(const std::shared_ptr<ov::Input<const ov::Node>>& input) {
    _input = input;
}

Napi::Object Input<const ov::Node>::Wrap(Napi::Env env, std::shared_ptr<ov::Input<const ov::Node>> input) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    Input* input_ptr = Napi::ObjectWrap<Input>::Unwrap(obj);
    input_ptr->set_input(input);
    return obj;
}

Napi::Value Input<const ov::Node>::get_shape(const Napi::CallbackInfo& info) {
    auto shape = _input->get_shape();
    return cpp_to_js<ov::Shape, Napi::Array>(info, shape);
}
