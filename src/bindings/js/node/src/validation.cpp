// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "node/include/validation.hpp"

bool check_object(const Napi::CallbackInfo& info, int index) {
    return info[index].ToObject();
}

bool check_tensor(const Napi::CallbackInfo& info, int index) {
    const auto& prototype = info.Env().GetInstanceData<AddonData>()->tensor;
    return info[index].ToObject().InstanceOf(prototype.Value().As<Napi::Function>());
}
