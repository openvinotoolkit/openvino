// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "low_precision/eltwise_base_transformation.hpp"

using namespace ::testing;
using namespace std;

using namespace ov;
using namespace ov::pass::low_precision;

class EltwiseTransformationIsBroadcastedTests : public ::testing::Test {
protected:
    const Shape c1 = Shape({ 1ul });
    const Shape c1000 = Shape({ 1000ul });
    const Shape n1c1 = Shape({ 1ul, 1ul });
    const Shape n1c256 = Shape({ 1ul, 256ul });
    const Shape n1c1000h1w1 = Shape({ 1ul, 1000ul, 1ul, 1ul });
    const Shape n1c32h144w144 = Shape({ 1ul, 32ul, 144ul, 144ul });
};

TEST_F(EltwiseTransformationIsBroadcastedTests, c1) {
    ASSERT_TRUE(EltwiseBaseTransformation::isBroadcasted(c1));
}

TEST_F(EltwiseTransformationIsBroadcastedTests, c1000) {
    ASSERT_FALSE(EltwiseBaseTransformation::isBroadcasted(c1000));
}

TEST_F(EltwiseTransformationIsBroadcastedTests, n1c1) {
    ASSERT_TRUE(EltwiseBaseTransformation::isBroadcasted(n1c1));
}

TEST_F(EltwiseTransformationIsBroadcastedTests, n1c256) {
    ASSERT_FALSE(EltwiseBaseTransformation::isBroadcasted(n1c256));
}

TEST_F(EltwiseTransformationIsBroadcastedTests, n1c1000h1w1) {
    ASSERT_TRUE(EltwiseBaseTransformation::isBroadcasted(n1c1000h1w1));
}

TEST_F(EltwiseTransformationIsBroadcastedTests, n1c32h144w144) {
    ASSERT_FALSE(EltwiseBaseTransformation::isBroadcasted(n1c32h144w144));
}

