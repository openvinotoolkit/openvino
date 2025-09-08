// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/node_output.hpp"

#include "node/include/addon.hpp"
#include "node/include/helper.hpp"
#include "node/include/partial_shape_wrap.hpp"

Output<ov::Node>::Output(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Output<ov::Node>>(info), _output{} {}

Napi::Function Output<ov::Node>::get_class(Napi::Env env) {
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

ov::Output<ov::Node> Output<ov::Node>::get_output() const {
    return _output;
}

Napi::Object Output<ov::Node>::wrap(Napi::Env env, ov::Output<ov::Node> output) {
    const auto& prototype = env.GetInstanceData<AddonData>()->output;
    if (!prototype) {
        OPENVINO_THROW("Invalid pointer to Output prototype.");
    }
    const auto& obj = prototype.New({});
    Output* output_ptr = Napi::ObjectWrap<Output>::Unwrap(obj);
    output_ptr->_output = output;
    return obj;
}

Napi::Value Output<ov::Node>::get_shape(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::Shape, Napi::Array>(info, _output.get_shape());
}

Napi::Value Output<ov::Node>::get_partial_shape(const Napi::CallbackInfo& info) {
    return PartialShapeWrap::wrap(info.Env(), _output.get_partial_shape());
}

Napi::Value Output<ov::Node>::get_any_name(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), _output.get_any_name());
}

Output<const ov::Node>::Output(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Output<const ov::Node>>(info),
      _output{} {}

Napi::Function Output<const ov::Node>::get_class(Napi::Env env) {
    return Output::DefineClass(
        env,
        "ConstOutput",
        {Output<const ov::Node>::InstanceMethod("getShape", &Output<const ov::Node>::get_shape),
         Output<const ov::Node>::InstanceAccessor<&Output<const ov::Node>::get_shape>("shape"),
         Output<const ov::Node>::InstanceMethod("getPartialShape", &Output<const ov::Node>::get_partial_shape),
         Output<const ov::Node>::InstanceMethod("getAnyName", &Output<const ov::Node>::get_any_name),
         Output<const ov::Node>::InstanceAccessor<&Output<const ov::Node>::get_any_name>("anyName"),
         Output<const ov::Node>::InstanceMethod("toString", &Output<const ov::Node>::get_any_name)});
}

ov::Output<const ov::Node> Output<const ov::Node>::get_output() const {
    return _output;
}

Napi::Object Output<const ov::Node>::wrap(Napi::Env env, ov::Output<const ov::Node> output) {
    const auto& prototype = env.GetInstanceData<AddonData>()->const_output;
    if (!prototype) {
        OPENVINO_THROW("Invalid pointer to ConstOutput prototype.");
    }
    const auto& obj = prototype.New({});
    Output* output_ptr = Napi::ObjectWrap<Output>::Unwrap(obj);
    output_ptr->_output = output;
    return obj;
}

Napi::Value Output<const ov::Node>::get_shape(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::Shape, Napi::Array>(info, _output.get_shape());
}

Napi::Value Output<const ov::Node>::get_partial_shape(const Napi::CallbackInfo& info) {
    return PartialShapeWrap::wrap(info.Env(), _output.get_partial_shape());
}

Napi::Value Output<const ov::Node>::get_any_name(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), _output.get_any_name());
}
