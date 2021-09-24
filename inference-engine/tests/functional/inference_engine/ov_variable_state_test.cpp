// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ie_blob.h"

#include <openvino/core/except.hpp>
#include <openvino/runtime/variable_state.hpp>

using namespace ::testing;
using namespace std;

TEST(VariableStateOVTests, throwsOnUninitializedReset) {
    ov::runtime::VariableState state;
    ASSERT_THROW(state.reset(), ov::Exception);
}

TEST(VariableStateOVTests, throwsOnUninitializedGetname) {
    ov::runtime::VariableState state;
    ASSERT_THROW(state.get_name(), ov::Exception);
}

TEST(VariableStateOVTests, throwsOnUninitializedGetState) {
    ov::runtime::VariableState state;
    ASSERT_THROW(state.get_state(), ov::Exception);
}

TEST(VariableStateOVTests, throwsOnUninitializedSetState) {
    ov::runtime::VariableState state;
    ov::Tensor tensor;
    ASSERT_THROW(state.set_state(tensor), ov::Exception);
}
