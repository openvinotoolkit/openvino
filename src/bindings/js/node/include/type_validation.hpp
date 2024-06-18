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
    OPENVINO_THROW("get_attr_type is not implemented for passed type!");
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
    OPENVINO_THROW("Validation for this type is not implemented!");
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

template <typename... Ts>
std::string get_signature() {
    if constexpr (sizeof...(Ts) == 0) {
        return "()";
    } else {
        std::string signature = "(";
        (signature.append(get_attr_type<Ts>()).append(", "), ...);
        signature.pop_back();
        signature.back() = ')';

        return signature;
    }
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
    const auto signature_attributes = get_signature<Ts...>();
    allowed_signatures.push_back(signature_attributes);

    return validate_detail<Ts...>(info);
};

std::string get_parameters_error_msg(const Napi::CallbackInfo& info, std::vector<std::string>& allowed_signatures);
}  // namespace js
}  // namespace ov
