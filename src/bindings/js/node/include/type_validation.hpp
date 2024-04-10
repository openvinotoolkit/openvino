// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include "node/include/addon.hpp"
#include "openvino/openvino.hpp"

/** @brief Checks if Napi::Value is a TensorWrap.*/
bool is_tensor(const Napi::Env& env, const Napi::Value& value);
