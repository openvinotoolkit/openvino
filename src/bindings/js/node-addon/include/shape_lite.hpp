// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <openvino/core/shape.hpp>

#include "napi.h"

class ShapeLite {
public:
    ShapeLite(const Napi::CallbackInfo& info);
    // ShapeLite(uintptr_t data, int dim);
    // ShapeLite(ov::Shape* shape);

    // uintptr_t get_data();
    // int get_dim();
    // int shape_size();
    // ov::Shape get_original();

    Napi::Number get_dim(const Napi::CallbackInfo& info);
    Napi::Number shape_size(const Napi::CallbackInfo& info);
    Napi::Value get_data(const Napi::CallbackInfo& info);

    //TO_DO
    static Napi::Function GetClassConstructor(Napi::Env env);
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Object Wrap(Napi::Env env, ov::Tensor tensor);



private:
    ov::Shape shape;
};
