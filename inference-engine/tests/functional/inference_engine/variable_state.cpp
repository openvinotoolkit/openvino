// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_infer_request.hpp>
#include "unit_test_utils/mocks/mock_ie_ivariable_state.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

IE_SUPPRESS_DEPRECATED_START

TEST(VariableStateCPPTests, throwsOnUninitialized) {
    std::shared_ptr<IVariableState> ptr;
    ASSERT_THROW(VariableState var(ptr), InferenceEngine::NotAllocated);
}

TEST(VariableStateCPPTests, nothrowOnInitialized) {
    std::shared_ptr<IVariableState> ptr = std::make_shared<MockIVariableState>();
    ASSERT_NO_THROW(VariableState var(ptr));
}

TEST(VariableStateCPPTests, throwsOnUninitializedGetLastState) {
    VariableState req;
    ASSERT_THROW(req.GetLastState(), InferenceEngine::NotAllocated);
}

IE_SUPPRESS_DEPRECATED_END

TEST(VariableStateCPPTests, throwsOnUninitializedReset) {
    VariableState req;
    ASSERT_THROW(req.Reset(), InferenceEngine::NotAllocated);
}

TEST(VariableStateCPPTests, throwsOnUninitializedGetname) {
    VariableState req;
    ASSERT_THROW(req.GetName(), InferenceEngine::NotAllocated);
}

TEST(VariableStateCPPTests, throwsOnUninitializedGetState) {
    VariableState req;
    ASSERT_THROW(req.GetState(), InferenceEngine::NotAllocated);
}

TEST(VariableStateCPPTests, throwsOnUninitializedSetState) {
    VariableState req;
    Blob::Ptr blob;
    ASSERT_THROW(req.SetState(blob), InferenceEngine::NotAllocated);
}
