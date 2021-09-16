// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/runtime/variable_state.hpp>

using namespace ::testing;
using namespace std;

TEST(VariableStateOVTests, throwsOnUninitializedReset) {
    ov::runtime::VariableState state;
    ASSERT_THROW(state.reset(), InferenceEngine::NotAllocated);
}

TEST(VariableStateOVTests, throwsOnUninitializedGetname) {
    ov::runtime::VariableState state;
    ASSERT_THROW(state.get_name(), InferenceEngine::NotAllocated);
}

TEST(VariableStateOVTests, throwsOnUninitializedGetState) {
    ov::runtime::VariableState state;
    ASSERT_THROW(state.get_state(), InferenceEngine::NotAllocated);
}

TEST(VariableStateOVTests, throwsOnUninitializedSetState) {
    ov::runtime::VariableState state;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(state.set_state(blob), InferenceEngine::NotAllocated);
}
