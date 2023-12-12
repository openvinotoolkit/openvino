// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

struct AddonData {
    Napi::FunctionReference* compiled_model_prototype;
    Napi::FunctionReference* core_prototype;
    Napi::FunctionReference* const_output_prototype;
    Napi::FunctionReference* infer_request_prototype;
    Napi::FunctionReference* model_prototype;
    Napi::FunctionReference* output_prototype;
    Napi::FunctionReference* partial_shape_prototype;
    Napi::FunctionReference* ppp_prototype;
    Napi::FunctionReference* tensor_prototype;
};
