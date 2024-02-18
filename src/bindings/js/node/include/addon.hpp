// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

typedef Napi::Function (*Prototype)(Napi::Env);

/** @brief A structure with data that will be associated with the instance of the ov.js node-addon. */
struct AddonData {
    Napi::FunctionReference compiled_model;
    Napi::FunctionReference core;
    Napi::FunctionReference const_output;
    Napi::FunctionReference infer_request;
    Napi::FunctionReference model;
    Napi::FunctionReference output;
    Napi::FunctionReference partial_shape;
    Napi::FunctionReference ppp;
    Napi::FunctionReference tensor;
};

void init_class(Napi::Env env,
                Napi::Object exports,
                std::string class_name,
                Prototype func,
                Napi::FunctionReference& reference);

Napi::Object init_module(Napi::Env env, Napi::Object exports);
