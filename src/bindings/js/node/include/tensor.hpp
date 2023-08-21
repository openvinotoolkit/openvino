// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief This is a header file for the NAPI POC TensorWrap
 * @file src/TensorWrap.hpp
 */
#pragma once

#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/tensor.hpp>

#include "element_type.hpp"
#include "errors.hpp"
#include "helper.hpp"
#include "napi.h"


class TensorWrap : public Napi::ObjectWrap<TensorWrap> {
public:
    /**
     * @brief Constructs TensorWrap class from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty or contain three params.
     * @param info[0] Exposed to JS enumElementType as ov.element
     * @param info[1] Array or Int32Array to create Shape
     * @param info[2] Float32Array with tensor data
     * @throw Exception if params are of invalid type.
     */
    TensorWrap(const Napi::CallbackInfo& info);

    /**
     * @brief Defines a Javascript Tensor class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript Tensor class.
     */
    static Napi::Function GetClassConstructor(Napi::Env env);
    
    /** @brief This method is called during initialization of OpenVino native add-on. 
     * It exports JavaScript Tensor class. 
     */
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    ov::Tensor get_tensor();
    void set_tensor(const ov::Tensor& tensor);
    /**
     * @brief Creates JavaScript Tensor object and wraps inside of it ov::Tensor object.
     * @param env The environment in which to construct a JavaScript object.
     * @param tensor ov::Tensor to wrap.
     * @return Javascript Tensor as Napi::Object. (Not TensorWrap object)
     */
    static Napi::Object Wrap(Napi::Env env, ov::Tensor tensor);

    /**
     * @brief Helper function to access the tensor data as an attribute of JavaScript Tensor.
     * @param info Contains information about the environment in which to create the Napi::Float32Array instance.
     * @return Napi::Float32Array containing the tensor data.
     */
    Napi::Value get_data(const Napi::CallbackInfo& info);

    /** @return A Javascript Shape object containing a tensor shape. */
    Napi::Value get_shape(const Napi::CallbackInfo& info);
    /** @return Napi::String containing ov::element type. */
    Napi::Value get_precision(const Napi::CallbackInfo& info);

private:
    ov::Tensor _tensor;
};
