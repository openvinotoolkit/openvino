// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/type_validation.hpp"

bool is_tensor(const Napi::Env& env, const Napi::Value& value) {
    const auto& prototype = env.GetInstanceData<AddonData>()->tensor;
    return value.ToObject().InstanceOf(prototype.Value().As<Napi::Function>());
}
