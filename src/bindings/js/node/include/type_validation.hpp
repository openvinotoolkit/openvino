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

template <class... Ts>
struct InputParameters {
    template <size_t... Is>
    static auto validate(const Napi::CallbackInfo& info, std::index_sequence<Is...>) {
        return sizeof...(Ts) == info.Length() && (... && (validate_value<Ts>(info.Env(), info[Is])));
    }
};

template <typename... Ts>
bool validate(const Napi::CallbackInfo& info) {
    return InputParameters<Ts...>::validate(info, std::index_sequence_for<Ts...>{});
};

template <typename... Ts>
bool validate(const Napi::CallbackInfo& info, std::vector<std::string>& allowed_signatures) {
    allowed_signatures.push_back(get_signature<Ts...>());

    return InputParameters<Ts...>::validate(info, std::index_sequence_for<Ts...>{});
};

std::string get_parameters_error_msg(const Napi::CallbackInfo& info, std::vector<std::string>& allowed_signatures);
}  // namespace js
}  // namespace ov
