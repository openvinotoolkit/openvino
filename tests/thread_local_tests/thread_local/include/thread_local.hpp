// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <iostream>
#include <string>

extern "C" __declspec(dllexport) void core_get_property_test(const std::string& target_device);
extern "C" __declspec(dllexport) void core_infer_test(const std::string& target_device);
