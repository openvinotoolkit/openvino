// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

struct AddonData {
    AddonData() {};
    Napi::FunctionReference* core_prototype;
    Napi::FunctionReference* model_prototype;
};
