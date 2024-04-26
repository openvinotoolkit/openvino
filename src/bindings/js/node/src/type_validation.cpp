// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/type_validation.hpp"

bool is_tensor(const Napi::Env& env, const Napi::Value& value) {
    const auto& prototype = env.GetInstanceData<AddonData>()->tensor;
    return value.ToObject().InstanceOf(prototype.Value().As<Napi::Function>());
}

namespace js {

template <>
bool validate_impl<std::string>(Napi::Value value) {
    return napi_string == value.Type();
}

template <>
bool validate_impl<int>(Napi::Value value) {
    return napi_number == value.Type();
}

template <>
bool validate_impl<ov::Model>(Napi::Value value) {
    return false;  // Env() neeeded.
}

template <>
bool validate_impl<Napi::Object>(Napi::Value value) {
    return napi_object == value.Type();
}

}  // namespace js
