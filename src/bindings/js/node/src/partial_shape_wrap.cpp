// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "partial_shape_wrap.hpp"

#include <iostream>

PartialShapeWrap::PartialShapeWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<PartialShapeWrap>(info) {
    const size_t attrs_length = info.Length();

    if (attrs_length == 0) {
        return;
    }

    if (attrs_length == 1 && info[0].IsString()) {
        std::string shape = std::string(info[0].ToString());

        _partial_shape = ov::PartialShape(shape);
        return;
    }

    reportError(info.Env(), "Cannot parse params");
}

Napi::Function PartialShapeWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "PartialShapeWrap",
                       {
                           InstanceMethod("isStatic", &PartialShapeWrap::is_static)
                       });
}

Napi::Object PartialShapeWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("PartialShape", func);
    return exports;
}

ov::PartialShape PartialShapeWrap::get_partial_shape() {
    return this->_partial_shape;
}

void PartialShapeWrap::set_partial_shape(const ov::PartialShape& partial_shape) {
    _partial_shape = partial_shape;
}

Napi::Object PartialShapeWrap::Wrap(Napi::Env env, ov::PartialShape partial_shape) {
    auto obj = GetClassConstructor(env).New({});
    auto t = Napi::ObjectWrap<PartialShapeWrap>::Unwrap(obj);
    t->set_partial_shape(partial_shape);
    
    return obj;
}

Napi::Value PartialShapeWrap::is_static(const Napi::CallbackInfo& info) {
  return cpp_to_js<bool, Napi::Boolean>(info, _partial_shape.is_static());
}
