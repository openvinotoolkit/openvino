// Copyright (C) ?
//
//

/**
 * @brief This is a header file for the NAPI POC helper functions
 *
 * @file src/helper.hpp
 */
#pragma once
#include <napi.h>

#include <map>
#include <openvino/core/type/element_type.hpp>
#include <openvino/openvino.hpp>
#include <variant>

#include "element_type.hpp"

typedef enum {
    js_array,
} js_type;

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

template <>
int32_t js_to_cpp<int32_t>(const Napi::CallbackInfo& info,
                           const size_t idx,
                           const std::vector<napi_types>& acceptable_types);

/// @brief  A template specialization for TargetType int32_t
template <>
int32_t js_to_cpp<int32_t>(const Napi::CallbackInfo& info,
                           const size_t idx,
                           const std::vector<napi_types>& acceptable_types);

/// @brief  A template specialization for TargetType std::vector<size_t>
template <>
std::vector<size_t> js_to_cpp<std::vector<size_t>>(const Napi::CallbackInfo& info,
                                                   const size_t idx,
                                                   const std::vector<napi_types>& acceptable_types);

/// @brief  A template specialization for TargetType std::string
template <>
std::string js_to_cpp<std::string>(const Napi::CallbackInfo& info,
                                   const size_t idx,
                                   const std::vector<napi_types>& acceptable_types);

/// @brief  A template specialization for TargetType ov::element::Type_T
template <>
ov::element::Type_t js_to_cpp<ov::element::Type_t>(const Napi::CallbackInfo& info,
                                                   const size_t idx,
                                                   const std::vector<napi_types>& acceptable_types);

/// @brief  A template specialization for TargetType ov::Layout
/// @param  acceptable_types ov::Layout can be created from a napi_string
template <>
ov::Layout js_to_cpp<ov::Layout>(const Napi::CallbackInfo& info,
                                 const size_t idx,
                                 const std::vector<napi_types>& acceptable_types);

/// @brief  A template specialization for TargetType ov::Shape
template <>
ov::Shape js_to_cpp<ov::Shape>(const Napi::CallbackInfo& info,
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

/// @brief  A template specialization for TargetType ov::element::Type_t and SourceType ov::element::Type_t
template <>
Napi::String cpp_to_js<ov::element::Type_t, Napi::String>(const Napi::CallbackInfo& info, const ov::element::Type_t type);

const std::map<std::string, ov::element::Type_t> element_type_map = {{"f32", ov::element::Type_t::f32},
                                                                     {"f64", ov::element::Type_t::f64},
                                                                     {"i32", ov::element::Type_t::i32},
                                                                     {"u32", ov::element::Type_t::u32}};
