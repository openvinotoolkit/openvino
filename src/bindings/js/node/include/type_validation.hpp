// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include "node/include/addon.hpp"
#include "node/include/model_wrap.hpp"
#include "node/include/tensor.hpp"
#include "openvino/openvino.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace js {
namespace NapiArg {
const char* get_type_name(napi_valuetype type);
}  // namespace NapiArg

std::string get_current_signature(const Napi::CallbackInfo& info);

template <typename T>
const char* get_attr_type() {
    throw std::runtime_error("get_attr_type is not implemented for passed type!");
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

template <typename T>
void get_signature_impl(std::vector<std::string>& attributes) {
    attributes.push_back(get_attr_type<T>());
};

template <typename T0, typename T1, typename... Ts>
void get_signature_impl(std::vector<std::string>& attributes) {
    attributes.push_back(get_attr_type<T0>());

    get_signature_impl<T1, Ts...>(attributes);
};

template <typename... Ts>
std::vector<std::string> get_signature() {
    std::vector<std::string> attributes;

    get_signature_impl<Ts...>(attributes);

    return attributes;
};

template <typename T>
bool validate_impl(const Napi::CallbackInfo& info, size_t depth) {
    return validate_value<T>(info.Env(), info[depth]);
};

template <typename T0, typename T1, typename... Ts>
bool validate_impl(const Napi::CallbackInfo& info, size_t depth) {
    bool is_passed = validate_value<T0>(info.Env(), info[depth]);

    if (!is_passed)
        return false;

    return validate_impl<T1, Ts...>(info, depth + 1);
};

template <typename... Ts>
bool validate_detail(const Napi::CallbackInfo& info) {
    const size_t attrs_length = info.Length();

    if (attrs_length != sizeof...(Ts))
        return false;

    return validate_impl<Ts...>(info, 0);
};

template <typename... Ts>
bool validate(const Napi::CallbackInfo& info) {
    return validate_detail<Ts...>(info);
};

template <typename... Ts>
bool validate(const Napi::CallbackInfo& info, std::vector<std::string>& allowed_signatures) {
    std::vector<std::string> signature_attributes = get_signature<Ts...>();
    allowed_signatures.push_back(std::string("(" + ov::util::join(signature_attributes) + ")"));

    return validate_detail<Ts...>(info);
};

std::string get_parameters_error_msg(const Napi::CallbackInfo& info, std::vector<std::string>& allowed_signatures);
}  // namespace js
}  // namespace ov
