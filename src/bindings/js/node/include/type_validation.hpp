// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include "node/include/addon.hpp"
#include "openvino/openvino.hpp"

/** @brief Checks if Napi::Value is a TensorWrap.*/
bool is_tensor(const Napi::Env& env, const Napi::Value& value);

namespace js {

template <typename T>
bool validate_impl(Napi::Value arg) {
    return false;
    // throw std::runtime_error("Validation for this type is not implemented!");
}

template <>
bool validate_impl<std::string>(Napi::Value value);

template <>
bool validate_impl<int>(Napi::Value value);
template <>
bool validate_impl<ov::Model>(Napi::Value value);

template <>
bool validate_impl<Napi::Object>(Napi::Value value);

// Forward declaration of the recursive case
template <typename T, typename... Args>
int validate_detail(const Napi::CallbackInfo& info, int index);

// Base case for a single type and a single argument
template <typename T>
bool validate_single(const Napi::CallbackInfo& info, int index) {
    validate_impl<T>(info[index]);
    return 0;
}

// Recursive case for multiple types and arguments
template <typename T, typename... Args>
int validate_detail(const Napi::CallbackInfo& info, int index) {
    validate_impl<T>(info[index]);
    // Call the next function in the chain
    return validate_single<Args...>(info, index + 1);
};

template <typename T>
int validate_detail(const Napi::CallbackInfo& info, int index) {
    // Call the base case function
    return validate_single<T>(info, index);
};

template <typename T, typename... Ts>
int validate(const Napi::CallbackInfo& info) {
    return validate_detail<T, Ts...>(info, 0);
};
}  // namespace js
