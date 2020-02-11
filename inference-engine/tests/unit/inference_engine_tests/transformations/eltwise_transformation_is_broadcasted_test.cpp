// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "low_precision_transformations/eltwise.hpp"

#include <ie_data.h>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class EltwiseTransformationIsBroadcastedTests : public ::testing::Test {
protected:
    const TensorDesc c1 = TensorDesc(Precision::FP32, { 1ul }, Layout::C);
    const TensorDesc c1000 = TensorDesc(Precision::FP32, { 1000ul }, Layout::C);
    const TensorDesc n1c1 = TensorDesc(Precision::FP32, { 1ul, 1ul }, Layout::NC);
    const TensorDesc n1c256 = TensorDesc(Precision::FP32, { 1ul, 256ul }, Layout::NC);
    const TensorDesc n1c1000h1w1 = TensorDesc(Precision::FP32, { 1ul, 1000ul, 1ul, 1ul }, Layout::NCHW);
    const TensorDesc n1c32h144w144 = TensorDesc(Precision::FP32, { 1ul, 32ul, 144ul, 144ul }, Layout::NCHW);
};

TEST_F(EltwiseTransformationIsBroadcastedTests, c1) {
    ASSERT_TRUE(EltwiseTransformation::isBroadcasted(c1));
}

TEST_F(EltwiseTransformationIsBroadcastedTests, c1000) {
    ASSERT_FALSE(EltwiseTransformation::isBroadcasted(c1000));
}

TEST_F(EltwiseTransformationIsBroadcastedTests, n1c1) {
    ASSERT_TRUE(EltwiseTransformation::isBroadcasted(n1c1));
}

TEST_F(EltwiseTransformationIsBroadcastedTests, n1c256) {
    ASSERT_FALSE(EltwiseTransformation::isBroadcasted(n1c256));
}

TEST_F(EltwiseTransformationIsBroadcastedTests, n1c1000h1w1) {
    ASSERT_TRUE(EltwiseTransformation::isBroadcasted(n1c1000h1w1));
}

TEST_F(EltwiseTransformationIsBroadcastedTests, n1c32h144w144) {
    ASSERT_FALSE(EltwiseTransformation::isBroadcasted(n1c32h144w144));
}

