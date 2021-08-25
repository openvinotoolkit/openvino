// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/runtime/remote_context.hpp>

using namespace ::testing;
using namespace std;

TEST(RemoteContextOVTests, throwsOnUninitializedReset) {
    ov::runtime::RemoteContext ctx;
    ASSERT_THROW(ctx.get_device_name(), InferenceEngine::NotAllocated);
}

TEST(RemoteContextOVTests, throwsOnUninitializedGetname) {
    ov::runtime::RemoteContext ctx;
    ASSERT_THROW(ctx.create_blob({}, {}), InferenceEngine::NotAllocated);
}

TEST(RemoteContextOVTests, throwsOnUninitializedGetParams) {
    ov::runtime::RemoteContext ctx;
    ASSERT_THROW(ctx.get_params(), InferenceEngine::NotAllocated);
}
