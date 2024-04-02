// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include "node/include/addon.hpp"
#include "openvino/openvino.hpp"

/** @brief Checks if Napi::CallbackInfo value at specified index is a TensorWrap.*/
bool check_tensor(const Napi::CallbackInfo& info, int index);
