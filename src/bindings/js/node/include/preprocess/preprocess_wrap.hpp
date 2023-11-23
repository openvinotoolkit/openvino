// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <openvino/openvino.hpp>

#include "preprocess/resize_algorithm.hpp"
#include "preprocess/pre_post_process_wrap.hpp"

class PreProcessWrap {
public:
    /** @brief This method is called during initialization of OpenVINO native add-on.
     * It exports JavaScript preprocess property.
     */
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    static Napi::Value PreProcessProperty(const Napi::CallbackInfo& info);
};
