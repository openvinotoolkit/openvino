// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/runtime/tensor.hpp"

class TensorWrap : public Napi::ObjectWrap<TensorWrap> {
public:
    /**
     * @brief Constructs TensorWrap class from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty or contain more arguments.
     * Two arguments are passed:
     * @param info[0] ov::element::Type as string or exposed to JS enumElementType.
     * @param info[1] ov::Shape as JS Array, Int32Array or Uint32Array
     * Three arguments are passed:
     * @param info[0] ov::element::Type as string or exposed to JS enumElementType.
     * @param info[1] ov::Shape as JS Array, Int32Array or Uint32Array
     * @param info[2] Tensor data as TypedArray
     * @throw Exception if params are of invalid type.
     */
    TensorWrap(const Napi::CallbackInfo& info);

    /**
     * @brief Defines a Javascript Tensor class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript Tensor class.
     */
    static Napi::Function get_class(Napi::Env env);

    ov::Tensor get_tensor() const;
    void set_tensor(const ov::Tensor& tensor);
    /**
     * @brief Creates JavaScript Tensor object and wraps inside of it ov::Tensor object.
     * @param env The environment in which to construct a JavaScript object.
     * @param tensor ov::Tensor to wrap.
     * @return Javascript Tensor as Napi::Object. (Not TensorWrap object)
     */
    static Napi::Object wrap(Napi::Env env, ov::Tensor tensor);

    /**
     * @brief Helper function to access the tensor data as an attribute of JavaScript Tensor.
     * @param info Contains information about the environment in which to create the Napi::TypedArray instance.
     * @return Napi::TypedArray containing the tensor data.
     */
    Napi::Value get_data(const Napi::CallbackInfo& info);

    /**
     * @brief Setter that fills underlaying Tensor's memory by copying data from TypedArray.
     * @throw Exception if data's size from TypedArray does not match the size of the tensor's data.
     */
    void set_data(const Napi::CallbackInfo& info, const Napi::Value& value);

    /** @return Napi::Array containing a tensor shape. */
    Napi::Value get_shape(const Napi::CallbackInfo& info);
    /** @return Napi::String containing ov::element type. */
    Napi::Value get_element_type(const Napi::CallbackInfo& info);
    /** @return Napi::Number containing tensor size as total number of elements. */
    Napi::Value get_size(const Napi::CallbackInfo& info);

private:
    ov::Tensor _tensor;
};
