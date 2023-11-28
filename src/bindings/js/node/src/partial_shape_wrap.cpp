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
        try {
            std::string shape = std::string(info[0].ToString());

            _partial_shape = ov::PartialShape(shape);
            return;
        } catch (std::exception& e) {
            reportError(info.Env(), e.what());
            return;
        }
    }

    reportError(info.Env(), "Cannot parse params");
}

Napi::Function PartialShapeWrap::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "PartialShapeWrap",
                       {
                           InstanceMethod("isStatic", &PartialShapeWrap::is_static),
                           InstanceMethod("isDynamic", &PartialShapeWrap::is_dynamic),
                           InstanceMethod("toString", &PartialShapeWrap::to_string),
                           InstanceMethod("getDimensions", &PartialShapeWrap::get_dimensions),
                       });
}

Napi::Object PartialShapeWrap::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);


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

Napi::Value PartialShapeWrap::is_dynamic(const Napi::CallbackInfo& info) {
    return cpp_to_js<bool, Napi::Boolean>(info, _partial_shape.is_dynamic());
}

Napi::Value PartialShapeWrap::to_string(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), _partial_shape.to_string());
}

Napi::Value PartialShapeWrap::get_dimensions(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::PartialShape, Napi::Array>(info, _partial_shape);
}
