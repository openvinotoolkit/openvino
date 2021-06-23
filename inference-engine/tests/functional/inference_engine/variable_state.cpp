// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_infer_request.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

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
