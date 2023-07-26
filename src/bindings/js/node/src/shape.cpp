// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "shape.hpp"

Shape::Shape(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Shape>(info) {
    if (info.Length() != 2 && info.Length() != 0)  // default contructor takes 0 args
        reportError(info.Env(), "Invalid number of arguments for Shape constructor.");
    else if (info.Length() == 2) {
        auto dim = info[0].As<Napi::Number>().Int32Value();
        auto data_array = info[1].As<Napi::Uint32Array>();
        for (int i = 0; i < dim; ++i)
            this->_shape.push_back(data_array[i]);
    }
}

Napi::Function Shape::GetClassConstructor(Napi::Env env) {
    return DefineClass(env,
                       "Shape",
                       {InstanceAccessor<&Shape::get_data>("data"),
                        InstanceMethod("getDim", &Shape::get_dim),
                        InstanceMethod("shapeSize", &Shape::shape_size),
                        InstanceMethod("getData", &Shape::get_data)});
}

Napi::Object Shape::Init(Napi::Env env, Napi::Object exports) {
    auto func = GetClassConstructor(env);

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Shape", func);
    return exports;
}

Napi::Object Shape::Wrap(Napi::Env env, ov::Shape shape) {
    Napi::HandleScope scope(env);
    auto obj = GetClassConstructor(env).New({});
    auto t = Napi::ObjectWrap<Shape>::Unwrap(obj);
    t->set_shape(shape);
    return obj;
}

Napi::Value Shape::get_data(const Napi::CallbackInfo& info) {
    auto arr = Napi::Array::New(info.Env(), _shape.size());
    for (size_t i = 0; i < _shape.size(); ++i)
        arr[i] = _shape[i];

    return arr;
}

Napi::Value Shape::get_dim(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), this->_shape.size());
}

Napi::Value Shape::shape_size(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), ov::shape_size(this->_shape));
}

void Shape::set_shape(const ov::Shape& shape) {
    _shape = shape;
}

ov::Shape Shape::get_original() {
    return this->_shape;
}
