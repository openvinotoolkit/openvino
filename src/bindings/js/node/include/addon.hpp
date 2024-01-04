// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

typedef Napi::Function (*Prototype)(Napi::Env);

/** @brief A structure with data that will be associated with the instance of the ov.js node-addon. */
struct AddonData {
    Napi::FunctionReference* compiled_model = new Napi::FunctionReference();
    Napi::FunctionReference* core = new Napi::FunctionReference();
    Napi::FunctionReference* const_output = new Napi::FunctionReference();
    Napi::FunctionReference* infer_request = new Napi::FunctionReference();
    Napi::FunctionReference* model = new Napi::FunctionReference();
    Napi::FunctionReference* output = new Napi::FunctionReference();
    Napi::FunctionReference* partial_shape = new Napi::FunctionReference();
    Napi::FunctionReference* ppp = new Napi::FunctionReference();
    Napi::FunctionReference* tensor = new Napi::FunctionReference();
};

void init_class(Napi::Env env,
                Napi::Object exports,
                std::string class_name,
                Prototype func,
                Napi::FunctionReference* reference);

Napi::Object init_all(Napi::Env env, Napi::Object exports);
