// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include "node/include/addon.hpp"
#include "node/include/model_wrap.hpp"
#include "node/include/tensor.hpp"
#include "openvino/openvino.hpp"
#include "openvino/util/common_util.hpp"

namespace NapiTypename {
static const char UNDEFINED_STR[] = "undefined";
static const char NULL_STR[] = "null";
static const char BOOLEAN_STR[] = "boolean";
static const char NUMBER_STR[] = "number";
static const char STIRNG_STR[] = "string";
static const char SYMBOL_STR[] = "symbol";
static const char OBJECT_STR[] = "object";
static const char FUNCTION_STR[] = "function";
static const char EXTERNAL_STR[] = "external";
static const char BIGINT_STR[] = "bigint";
static const char UNKNOWN_STR[] = "unknown";
}  // namespace NapiTypename

namespace BindingTypename {
static const char INT[] = "Integer";
static const char MODEL[] = "Model";
static const char TENSOR[] = "Tensor";
static const char BUFFER[] = "Buffer";
}  // namespace BindingTypename

namespace NapiArg {
const char* get_type_name(napi_valuetype type);
}  // namespace NapiArg

namespace js {
std::string get_current_signature(const Napi::CallbackInfo& info);

template <typename T>
const char* get_attr_type() {
    throw std::runtime_error("get_attr_type 123 is not implemented for passed type!");
};

template <>
const char* get_attr_type<Napi::String>();

template <>
const char* get_attr_type<Napi::Object>();

template <>
const char* get_attr_type<Napi::Buffer<uint8_t>>();

template <>
const char* get_attr_type<int>();

template <>
const char* get_attr_type<ModelWrap>();

template <>
const char* get_attr_type<TensorWrap>();

template <typename T>
bool validate_value(const Napi::Env& env, const Napi::Value& arg) {
    throw std::runtime_error("Validation for this type is not implemented!");
};

template <>
bool validate_value<Napi::String>(const Napi::Env& env, const Napi::Value& value);

template <>
bool validate_value<Napi::Object>(const Napi::Env& env, const Napi::Value& value);

template <>
bool validate_value<Napi::Buffer<uint8_t>>(const Napi::Env& env, const Napi::Value& value);

template <>
bool validate_value<int>(const Napi::Env& env, const Napi::Value& value);

template <>
bool validate_value<ModelWrap>(const Napi::Env& env, const Napi::Value& value);

/** @brief Checks if Napi::Value is a Tensor.*/
template <>
bool validate_value<TensorWrap>(const Napi::Env& env, const Napi::Value& value);

template <typename Arg>
std::vector<std::string> get_signature() {
    std::vector<std::string> attributes;

    attributes.push_back(get_attr_type<Arg>());

    return attributes;
};
template <typename Arg, typename Arg1>
std::vector<std::string> get_signature() {
    std::vector<std::string> attributes;

    attributes.push_back(get_attr_type<Arg>());
    attributes.push_back(get_attr_type<Arg1>());

    return attributes;
};
template <typename Arg, typename Arg1, typename Arg2>
std::vector<std::string> get_signature() {
    std::vector<std::string> attributes;

    attributes.push_back(get_attr_type<Arg>());
    attributes.push_back(get_attr_type<Arg1>());
    attributes.push_back(get_attr_type<Arg2>());

    return attributes;
};

template <typename Arg, typename Arg1, typename Arg2>
bool validate_detail(const Napi::CallbackInfo& info) {
    return info.Length() == 3 && validate_value<Arg>(info.Env(), info[0]) &&
           validate_value<Arg1>(info.Env(), info[1]) && validate_value<Arg2>(info.Env(), info[2]);
};

template <typename Arg, typename Arg1>
bool validate_detail(const Napi::CallbackInfo& info) {
    return info.Length() == 2 && validate_value<Arg>(info.Env(), info[0]) && validate_value<Arg1>(info.Env(), info[1]);
};

template <typename Arg>
bool validate_detail(const Napi::CallbackInfo& info) {
    return info.Length() == 1 && validate_value<Arg>(info.Env(), info[0]);
};

template <typename T, typename... Ts>
bool validate(const Napi::CallbackInfo& info) {
    return validate_detail<T, Ts...>(info);
};

template <typename T, typename... Ts>
bool validate(const Napi::CallbackInfo& info, std::vector<std::string>& checked_signatures) {
    std::vector<std::string> signature_attributes = get_signature<T, Ts...>();
    checked_signatures.push_back(std::string("(" + ov::util::join(signature_attributes) + ")"));

    return validate_detail<T, Ts...>(info);
};

std::string get_parameters_error_msg(const Napi::CallbackInfo& info, std::vector<std::string>& checked_signatures);
}  // namespace js
