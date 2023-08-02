// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <iostream>
#include <openvino/core/shape.hpp>

#include "errors.hpp"
#include "napi.h"

class Shape : public Napi::ObjectWrap<Shape> {
public:
    Shape(const Napi::CallbackInfo& info);

    ov::Shape get_original();

    static Napi::Function GetClassConstructor(Napi::Env env);
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Object Wrap(Napi::Env env, ov::Shape tensor);

    Napi::Value get_dim(const Napi::CallbackInfo& info);
    Napi::Value shape_size(const Napi::CallbackInfo& info);
    Napi::Value get_data(const Napi::CallbackInfo& info);

    void set_shape(const ov::Shape&);

private:
    ov::Shape _shape;
};
