// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Regression tests for the null-guard added to DispatchDataFunc::operator().
// Before the fix, calling a DispatchDataFunc constructed from nullptr would
// invoke a null std::function and throw std::bad_function_call.

#include "test_utils.h"
#include "common_utils/kernel_generator_base.hpp"

using namespace ov::intel_gpu;

TEST(dispatch_data_func, null_func_does_not_crash) {
    DispatchDataFunc func{nullptr};
    KernelData kd;
    RuntimeParams params;

    ASSERT_NO_THROW(func(params, kd, nullptr));
}

TEST(dispatch_data_func, valid_func_is_called) {
    bool called = false;
    DispatchDataFunc func{[&called](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {
        called = true;
    }};
    KernelData kd;
    RuntimeParams params;

    func(params, kd, nullptr);
    ASSERT_TRUE(called);
}

TEST(dispatch_data_func, default_constructed_is_null_safe) {
    // Default-constructed KernelData has update_dispatch_data_func{nullptr}.
    KernelData kd;
    RuntimeParams params;

    ASSERT_NO_THROW(kd.update_dispatch_data_func(params, kd, nullptr));
}
