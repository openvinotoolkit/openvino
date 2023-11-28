// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node_output.hpp"
#include "partial_shape_wrap.hpp"

#include "helper.hpp"

Output<ov::Node>::Output(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Output<ov::Node>>(info) {}

Napi::Function Output<ov::Node>::GetClassConstructor(Napi::Env env) {
    return Output::DefineClass(
        env,
        "Output",
        {Output<ov::Node>::InstanceMethod("getShape", &Output<ov::Node>::get_shape),
         Output<ov::Node>::InstanceAccessor<&Output<ov::Node>::get_shape>("shape"),
         Output<ov::Node>::InstanceMethod("getPartialShape", &Output<ov::Node>::get_partial_shape),
         Output<ov::Node>::InstanceMethod("getAnyName", &Output<ov::Node>::get_any_name),
         Output<ov::Node>::InstanceAccessor<&Output<ov::Node>::get_any_name>("anyName"),
         Output<ov::Node>::InstanceMethod("toString", &Output<ov::Node>::get_any_name)});
}

Napi::Object Output<ov::Node>::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);


    exports.Set("Output", func);
    return exports;
}

ov::Output<ov::Node> Output<ov::Node>::get_output() const {
    return _output;
}

Napi::Object Output<ov::Node>::Wrap(Napi::Env env, ov::Output<ov::Node> output) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    Output* output_ptr = Napi::ObjectWrap<Output>::Unwrap(obj);
    output_ptr->_output = output;
    return obj;
}

Napi::Value Output<ov::Node>::get_shape(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::Shape, Napi::Array>(info, _output.get_shape());
}

Napi::Value Output<ov::Node>::get_partial_shape(const Napi::CallbackInfo& info) {
    return PartialShapeWrap::Wrap(info.Env(), _output.get_partial_shape());
}

Napi::Value Output<ov::Node>::get_any_name(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), _output.get_any_name());
}

Output<const ov::Node>::Output(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Output<const ov::Node>>(info) {}

Napi::Function Output<const ov::Node>::GetClassConstructor(Napi::Env env) {
    return Output::DefineClass(
        env,
        "Output",
        {Output<const ov::Node>::InstanceMethod("getShape", &Output<const ov::Node>::get_shape),
         Output<const ov::Node>::InstanceAccessor<&Output<const ov::Node>::get_shape>("shape"),
         Output<const ov::Node>::InstanceMethod("getPartialShape", &Output<const ov::Node>::get_partial_shape),
         Output<const ov::Node>::InstanceMethod("getAnyName", &Output<const ov::Node>::get_any_name),
         Output<const ov::Node>::InstanceAccessor<&Output<const ov::Node>::get_any_name>("anyName"),
         Output<const ov::Node>::InstanceMethod("toString", &Output<const ov::Node>::get_any_name)});
}

Napi::Object Output<const ov::Node>::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    exports.Set("Output", func);
    return exports;
}

ov::Output<const ov::Node> Output<const ov::Node>::get_output() const {
    return _output;
}

Napi::Object Output<const ov::Node>::Wrap(Napi::Env env, ov::Output<const ov::Node> output) {
    Napi::HandleScope scope(env);
    Napi::Object obj = GetClassConstructor(env).New({});
    Output* output_ptr = Napi::ObjectWrap<Output>::Unwrap(obj);
    output_ptr->_output = output;
    return obj;
}

Napi::Value Output<const ov::Node>::get_shape(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::Shape, Napi::Array>(info, _output.get_shape());
}

Napi::Value Output<const ov::Node>::get_partial_shape(const Napi::CallbackInfo& info) {
    return PartialShapeWrap::Wrap(info.Env(), _output.get_partial_shape());
}

Napi::Value Output<const ov::Node>::get_any_name(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), _output.get_any_name());
}
