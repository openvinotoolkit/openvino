// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include <unordered_set>
#include <variant>

#include "openvino/core/type/element_type.hpp"
#include "openvino/openvino.hpp"

typedef enum {
    js_array,
} js_type;

const std::vector<std::string>& get_supported_types();

typedef std::variant<napi_valuetype, napi_typedarray_type, js_type> napi_types;

/**
 * @brief  Template function to convert Javascript data types into C++ data types
 * @tparam TargetType destinated C++ data type
 * @param info Napi::CallbackInfo contains all arguments passed to a function or method
 * @param idx specifies index of a argument inside info.
 * @param acceptable_types specifies napi types from which TargetType can be created
 * @return specified argument converted to a TargetType.
 */
template <typename TargetType>
TargetType js_to_cpp(const Napi::CallbackInfo& info, const size_t idx, const std::vector<napi_types>& acceptable_types);

template <typename TargetType>
TargetType js_to_cpp(const Napi::Value&, const std::vector<napi_types>& acceptable_types);

template <>
int32_t js_to_cpp<int32_t>(const Napi::CallbackInfo& info,
                           const size_t idx,
                           const std::vector<napi_types>& acceptable_types);

/** @brief  A template specialization for TargetType int32_t */
template <>
int32_t js_to_cpp<int32_t>(const Napi::CallbackInfo& info,
                           const size_t idx,
                           const std::vector<napi_types>& acceptable_types);

/** @brief  A template specialization for TargetType std::vector<size_t> */
template <>
std::vector<size_t> js_to_cpp<std::vector<size_t>>(const Napi::CallbackInfo& info,
                                                   const size_t idx,
                                                   const std::vector<napi_types>& acceptable_types);

/** @brief  A template specialization for TargetType std::unordered_set<std::string> */
template <>
std::unordered_set<std::string> js_to_cpp<std::unordered_set<std::string>>(
    const Napi::CallbackInfo& info,
    const size_t idx,
    const std::vector<napi_types>& acceptable_types);

/** @brief  A template specialization for TargetType std::string */
template <>
std::string js_to_cpp<std::string>(const Napi::CallbackInfo& info,
                                   const size_t idx,
                                   const std::vector<napi_types>& acceptable_types);

/** @brief  A template specialization for TargetType ov::element::Type_T */
template <>
ov::element::Type_t js_to_cpp<ov::element::Type_t>(const Napi::CallbackInfo& info,
                                                   const size_t idx,
                                                   const std::vector<napi_types>& acceptable_types);

/**
 * @brief  A template specialization for TargetType ov::Layout
 * @param  acceptable_types ov::Layout can be created from a napi_string
 */
template <>
ov::Layout js_to_cpp<ov::Layout>(const Napi::CallbackInfo& info,
                                 const size_t idx,
                                 const std::vector<napi_types>& acceptable_types);

/** @brief  A template specialization for TargetType ov::Shape */
template <>
ov::Shape js_to_cpp<ov::Shape>(const Napi::CallbackInfo& info,
                               const size_t idx,
                               const std::vector<napi_types>& acceptable_types);

/** @brief  A template specialization for TargetType ov::preprocess::ResizeAlgorithm */
template <>
ov::preprocess::ResizeAlgorithm js_to_cpp<ov::preprocess::ResizeAlgorithm>(
    const Napi::CallbackInfo& info,
    const size_t idx,
    const std::vector<napi_types>& acceptable_types);

/** @brief  A template specialization for TargetType ov::Any */
template <>
ov::Any js_to_cpp<ov::Any>(const Napi::Value&, const std::vector<napi_types>& acceptable_types);

/** @brief  A template specialization for TargetType std::map<std::string, ov::Any */
template <>
std::map<std::string, ov::Any> js_to_cpp<std::map<std::string, ov::Any>>(
    const Napi::CallbackInfo& info,
    const size_t idx,
    const std::vector<napi_types>& acceptable_types);

/**
 * @brief  Template function to convert C++ data types into Javascript data types
 * @tparam TargetType Destinated Javascript data type.
 * @tparam SourceType C++ data type.
 * @param info Contains the environment in which to construct a JavaScript object.
 * @return SourceType converted to a TargetType.
 */
template <typename SourceType, typename TargetType>
TargetType cpp_to_js(const Napi::CallbackInfo& info, SourceType);

/** @brief  A template specialization for TargetType ov::element::Type_t and SourceType ov::element::Type_t */
template <>
Napi::String cpp_to_js<ov::element::Type_t, Napi::String>(const Napi::CallbackInfo& info,
                                                          const ov::element::Type_t type);

template <>
Napi::Array cpp_to_js<ov::Shape, Napi::Array>(const Napi::CallbackInfo& info, const ov::Shape shape);

template <>
Napi::Array cpp_to_js<ov::PartialShape, Napi::Array>(const Napi::CallbackInfo& info, const ov::PartialShape shape);

template <>
Napi::Array cpp_to_js<ov::Dimension, Napi::Array>(const Napi::CallbackInfo& info, const ov::Dimension dim);

template <>
Napi::Boolean cpp_to_js<bool, Napi::Boolean>(const Napi::CallbackInfo& info, const bool value);

/** @brief Takes Napi::Value and parse Napi::Array or Napi::Object to ov::TensorVector. */
ov::TensorVector parse_input_data(const Napi::Value& input);

/** @brief Gets an input/output tensor from InferRequest by key. */
ov::Tensor get_request_tensor(ov::InferRequest& infer_request, const std::string key);

/** @brief Gets an input tensor from InferRequest by index. */
ov::Tensor get_request_tensor(ov::InferRequest& infer_request, const size_t idx);

/** @brief Creates ov::tensor from TensorWrap Object */
ov::Tensor cast_to_tensor(const Napi::Value& value);

/** @brief Creates ov::tensor from Napi::CallbackInfo value at specified index. */
ov::Tensor cast_to_tensor(const Napi::CallbackInfo& info, int index);

/** @brief Creates ov::tensor from TypedArray using given shape and element type*/
ov::Tensor cast_to_tensor(const Napi::TypedArray& data, const ov::Shape& shape, const ov::element::Type_t& type);

/** @brief A helper function to create a ov::Tensor from Napi::Value.
 * @param value a Napi::Value that can be either a TypedArray or a TensorWrap Object.
 * @param infer_request The reference to InferRequest.
 * @param key of the tensor to get from InferRequest.
 * @return ov::Tensor
 */
template <typename KeyType>
ov::Tensor value_to_tensor(const Napi::Value& value, ov::InferRequest& infer_request, const KeyType key) {
    if (value.IsTypedArray()) {
        const auto input = get_request_tensor(infer_request, key);
        const auto& shape = input.get_shape();
        const auto& type = input.get_element_type();
        const auto data = value.As<Napi::TypedArray>();
        return cast_to_tensor(data, shape, type);

    } else {
        return cast_to_tensor(value.As<Napi::Value>());
    }
}

napi_types napiType(const Napi::Value& val);

bool acceptableType(const Napi::Value& val, const std::vector<napi_types>& acceptable);

Napi::Value any_to_js(const Napi::CallbackInfo& info, ov::Any value);

ov::Any js_to_any(const Napi::CallbackInfo& info, Napi::Value value);

bool is_napi_value_int(const Napi::CallbackInfo& info, Napi::Value& num);
