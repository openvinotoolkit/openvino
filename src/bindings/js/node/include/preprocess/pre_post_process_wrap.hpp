// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/core/preprocess/pre_post_process.hpp"

class PrePostProcessorWrap : public Napi::ObjectWrap<PrePostProcessorWrap> {
public:
    /**
     * @brief Constructs PrePostProcessorWrap class from the Napi::CallbackInfo.
     * @param info //TO DO
     */
    PrePostProcessorWrap(const Napi::CallbackInfo& info);
    /**
     * @brief Defines a Javascript PrePostProcessor class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript PrePostProcessor class.
     */
    static Napi::Function get_class(Napi::Env env);

    Napi::Value input(const Napi::CallbackInfo& info);

    Napi::Value output(const Napi::CallbackInfo& info);

    void build(const Napi::CallbackInfo& info);

private:
    std::unique_ptr<ov::preprocess::PrePostProcessor> _ppp;
};
