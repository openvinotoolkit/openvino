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
 * @brief Gets corresponding ov::element::Type_t to the passed TypedArray type
 * @throw Exception if there is no corresponing or supported ov::element::Type_t
 */
const ov::element::Type_t& get_ov_type(napi_typedarray_type type);

/**
 * @brief  Template function to convert Javascript data types into C++ data types
 * @tparam TargetType destinated C++ data type
 * @param info Napi::CallbackInfo contains all arguments passed to a function or method
 * @param idx specifies index of a argument inside info.
 * @return specified argument converted to a TargetType.
 */
template <typename TargetType>
TargetType js_to_cpp(const Napi::Env& env, const Napi::Value& value);

template <typename TargetType>
TargetType js_to_cpp(const Napi::CallbackInfo& info, const size_t idx);

/** @brief  A template specialization for TargetType std::vector<size_t> */
template <>
std::vector<size_t> js_to_cpp<std::vector<size_t>>(const Napi::CallbackInfo& info, const size_t idx);

/** @brief  A template specialization for TargetType std::unordered_set<std::string> */
template <>
std::unordered_set<std::string> js_to_cpp<std::unordered_set<std::string>>(const Napi::CallbackInfo& info,
                                                                           const size_t idx);

/** @brief  A template specialization for TargetType std::string */
template <>
std::string js_to_cpp<std::string>(const Napi::CallbackInfo& info, const size_t idx);

/** @brief  A template specialization for TargetType ov::element::Type_T */
template <>
ov::element::Type_t js_to_cpp<ov::element::Type_t>(const Napi::CallbackInfo& info, const size_t idx);

/**
 * @brief  A template specialization for TargetType ov::Layout
 * ov::Layout has to be created from a napi_string
 */
template <>
ov::Layout js_to_cpp<ov::Layout>(const Napi::CallbackInfo& info, const size_t idx);

/** @brief  A template specialization for TargetType ov::Shape */
template <>
ov::Shape js_to_cpp<ov::Shape>(const Napi::CallbackInfo& info, const size_t idx);

/** @brief  A template specialization for TargetType ov::preprocess::ResizeAlgorithm */
template <>
ov::preprocess::ResizeAlgorithm js_to_cpp<ov::preprocess::ResizeAlgorithm>(const Napi::CallbackInfo& info,
                                                                           const size_t idx);

/** @brief  A template specialization for TargetType ov::Any */
template <>
ov::Any js_to_cpp<ov::Any>(const Napi::Env& env, const Napi::Value& value);

/** @brief  A template specialization for TargetType std::map<std::string, ov::Any
 * std::map<std::string, ov::Any> has to be created from napi_object
 */
template <>
std::map<std::string, ov::Any> js_to_cpp<std::map<std::string, ov::Any>>(const Napi::CallbackInfo& info,
                                                                         const size_t idx);

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

/**
 * @brief Creates JavaScript Model and wraps ov::Model inside of it.
 * @return Javascript Model as Napi::Object. (Not ModelWrap object)
 */
Napi::Object cpp_to_js(const Napi::Env& env, std::shared_ptr<ov::Model> model);

template <>
Napi::Boolean cpp_to_js<bool, Napi::Boolean>(const Napi::CallbackInfo& info, const bool value);

Napi::Object cpp_to_js(const Napi::Env& env, const ov::CompiledModel& compiled_model);

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

void fill_tensor_from_strings(ov::Tensor& tensor, const Napi::Array& arr);

/** @brief Helper method to create ov::tensor of ov::element::string type */
ov::Tensor cast_to_tensor(const Napi::Array& array);

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
        return cast_to_tensor(value.As<Napi::TypedArray>(), shape, type);
    } else if (value.IsArray()) {
        return cast_to_tensor(value.As<Napi::Array>());
    } else {
        return cast_to_tensor(value.As<Napi::Value>());
    }
}

napi_types napiType(const Napi::Value& val);

Napi::Value any_to_js(const Napi::CallbackInfo& info, ov::Any value);

bool is_napi_value_int(const Napi::Env& env, const Napi::Value& num);

ov::AnyMap to_anyMap(const Napi::Env&, const Napi::Value&);

std::string buffer_to_string(const Napi::Value& value);
