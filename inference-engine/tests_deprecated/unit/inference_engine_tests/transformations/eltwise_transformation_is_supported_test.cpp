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

class EltwiseTransformationIsSupportedTests : public ::testing::Test {
protected:
    const TensorDesc n1c1000h1w1 = TensorDesc(Precision::FP32, { 1ul, 1000ul, 1ul, 1ul }, Layout::NCHW);
    const TensorDesc n1c2000h1w1 = TensorDesc(Precision::FP32, { 1ul, 1000ul, 1ul, 1ul }, Layout::NCHW);
    const TensorDesc n1c1000 = TensorDesc(Precision::FP32, { 1ul, 1000ul }, Layout::NC);
    const TensorDesc n1c1 = TensorDesc(Precision::FP32, { 1ul, 1ul }, Layout::NC);
    const TensorDesc n1c2000 = TensorDesc(Precision::FP32, { 1ul, 2000ul }, Layout::NC);
    const TensorDesc c1 = TensorDesc(Precision::FP32, { 1ul }, Layout::C);
    const TensorDesc c1000 = TensorDesc(Precision::FP32, { 1000ul }, Layout::C);
    const TensorDesc c2000 = TensorDesc(Precision::FP32, { 2000ul }, Layout::C);
};

TEST_F(EltwiseTransformationIsSupportedTests, n1c1000h1w1_and_n1c2000h1w1) {
    ASSERT_TRUE(EltwiseTransformation::isSupported(n1c1000h1w1, n1c2000h1w1));
}

TEST_F(EltwiseTransformationIsSupportedTests, n1c1000h1w1_and_n1c1000h1w1) {
    ASSERT_TRUE(EltwiseTransformation::isSupported(n1c1000h1w1, n1c1000h1w1));
}

TEST_F(EltwiseTransformationIsSupportedTests, n1c1000h1w1_and_n1c1000) {
    ASSERT_TRUE(EltwiseTransformation::isSupported(n1c1000h1w1, n1c1000));
}

TEST_F(EltwiseTransformationIsSupportedTests, n1c1000h1w1_and_n1c2000) {
    ASSERT_FALSE(EltwiseTransformation::isSupported(n1c1000h1w1, n1c2000));
}

TEST_F(EltwiseTransformationIsSupportedTests, n1c1000h1w1_and_c1) {
    ASSERT_TRUE(EltwiseTransformation::isSupported(n1c1000h1w1, c1));
}

TEST_F(EltwiseTransformationIsSupportedTests, n1c1000h1w1_and_c1000) {
    ASSERT_TRUE(EltwiseTransformation::isSupported(n1c1000h1w1, c1000));
}

TEST_F(EltwiseTransformationIsSupportedTests, n1c1000h1w1_and_c2000) {
    ASSERT_FALSE(EltwiseTransformation::isSupported(n1c1000h1w1, c2000));
}

TEST_F(EltwiseTransformationIsSupportedTests, n1c1000_and_n1c1000) {
    ASSERT_TRUE(EltwiseTransformation::isSupported(n1c1000, n1c1000));
}

TEST_F(EltwiseTransformationIsSupportedTests, n1c1000_and_n1c2000) {
    ASSERT_FALSE(EltwiseTransformation::isSupported(n1c1000, n1c2000));
}

TEST_F(EltwiseTransformationIsSupportedTests, n1c2000h1w1_and_n1c1000) {
    ASSERT_TRUE(EltwiseTransformation::isSupported(n1c2000h1w1, n1c1000));
}

TEST_F(EltwiseTransformationIsSupportedTests, n1c1000_and_n1c1) {
    ASSERT_TRUE(EltwiseTransformation::isSupported(n1c1000, n1c1));
}
