// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/core/partial_shape.hpp"

class PartialShapeWrap : public Napi::ObjectWrap<PartialShapeWrap> {
public:
    /**
     * @brief Constructs PartialShapeWrap from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty.
     */
    PartialShapeWrap(const Napi::CallbackInfo& info);
    /**
     * @brief Defines a Javascript PartialShape class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript PartialShape class.
     */
    static Napi::Function get_class(Napi::Env env);

    /**
     * @brief Creates JavaScript PartialShape object and wraps inside of it ov::PartialShape object.
     * @param env The environment in which to construct a JavaScript object.
     * @param partial_shape ov::PartialShape to wrap.
     * @return Javascript PartialShape as Napi::Object. (Not PartialShapeWrap object)
     */
    static Napi::Object wrap(Napi::Env env, ov::PartialShape partial_shape);

    Napi::Value is_static(const Napi::CallbackInfo& info);
    Napi::Value is_dynamic(const Napi::CallbackInfo& info);
    Napi::Value to_string(const Napi::CallbackInfo& info);
    Napi::Value get_dimensions(const Napi::CallbackInfo& info);

private:
    ov::PartialShape _partial_shape;
};
