// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>
#include <openvino/core/partial_shape.hpp>

#include "errors.hpp"
#include "helper.hpp"

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
    static Napi::Function GetClassConstructor(Napi::Env env);
    /** @brief This method is called during initialization of OpenVino native add-on.
     * It exports JavaScript PartialShape class.
     */
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    ov::PartialShape get_partial_shape();
    void set_partial_shape(const ov::PartialShape& partial_shape);
    /**
     * @brief Creates JavaScript PartialShape object and wraps inside of it ov::PartialShape object.
     * @param env The environment in which to construct a JavaScript object.
     * @param partial_shape ov::PartialShape to wrap.
     * @return Javascript PartialShape as Napi::Object. (Not PartialShapeWrap object)
     */
    static Napi::Object Wrap(Napi::Env env, ov::PartialShape partial_shape);

    Napi::Value is_static(const Napi::CallbackInfo& info);
    Napi::Value is_dynamic(const Napi::CallbackInfo& info);
    Napi::Value to_string(const Napi::CallbackInfo& info);

private:
    ov::PartialShape _partial_shape;       
};
