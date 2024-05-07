// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include "node/include/addon.hpp"
#include "openvino/openvino.hpp"

namespace js {

template <typename T>
bool validate_value(const Napi::Env& env, Napi::Value arg) {
    throw std::runtime_error("Validation for this type is not implemented!");
}

template <>
bool validate_value<std::string>(const Napi::Env& env, Napi::Value value);

template <>
bool validate_value<int>(const Napi::Env& env, Napi::Value value);

template <>
bool validate_value<ov::Model>(const Napi::Env& env, Napi::Value value);

/** @brief Checks if Napi::Value is a Tensor.*/
template <>
bool validate_value<ov::Tensor>(const Napi::Env& env, Napi::Value value);

template <>
bool validate_value<Napi::Object>(const Napi::Env& env, Napi::Value value);

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
}  // namespace js
