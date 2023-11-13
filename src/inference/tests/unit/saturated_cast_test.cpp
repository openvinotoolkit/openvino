// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "ie_precision.hpp"
#include "precision_utils.h"

#ifdef USE_OPENCV

#    include <opencv2/core.hpp>

using namespace InferenceEngine;

class SaturateCastTestsI64ToI32 : public ov::test::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::I64>::value_type;
    using toType = typename PrecisionTrait<Precision::I32>::value_type;
};

TEST_F(SaturateCastTestsI64ToI32, I64ToI32NonNarrowingPositive) {
    const auto value = fromType{42};
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

TEST_F(SaturateCastTestsI64ToI32, I64ToI32NonNarrowingNegative) {
    const auto value = fromType{-42};
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

TEST_F(SaturateCastTestsI64ToI32, I64ToI32NarrowingMaxToMax) {
    const auto value = std::numeric_limits<fromType>::max();
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

TEST_F(SaturateCastTestsI64ToI32, I64ToI32NarrowingNonMaxToMax) {
    const auto value = std::numeric_limits<fromType>::max() - 1;
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

TEST_F(SaturateCastTestsI64ToI32, I64ToI32NarrowingMinToMin) {
    const auto value = std::numeric_limits<fromType>::min();
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

TEST_F(SaturateCastTestsI64ToI32, I64ToI32NarrowingNonMinToMin) {
    const auto value = std::numeric_limits<fromType>::min() + 1;
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

class SaturateCastTestsU64ToI32 : public ov::test::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::U64>::value_type;
    using toType = typename PrecisionTrait<Precision::I32>::value_type;
};

TEST_F(SaturateCastTestsU64ToI32, U64ToI32NonNarrowing) {
    const auto value = fromType{42};
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

TEST_F(SaturateCastTestsU64ToI32, U64ToI32NarrowingMaxToMax) {
    const auto value = std::numeric_limits<fromType>::max();
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

TEST_F(SaturateCastTestsU64ToI32, U64ToI32NarrowingNonMaxToMax) {
    const auto value = std::numeric_limits<fromType>::max() - 1;
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

class SaturateCastTestsBoolToU8 : public ov::test::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::BOOL>::value_type;
    using toType = typename PrecisionTrait<Precision::U8>::value_type;
};

TEST_F(SaturateCastTestsBoolToU8, BOOLtoU8MaxToNonMax) {
    const auto value = std::numeric_limits<fromType>::max();
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

class SaturateCastTestsBoolToI32 : public ov::test::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::BOOL>::value_type;
    using toType = typename PrecisionTrait<Precision::I32>::value_type;
};

TEST_F(SaturateCastTestsBoolToI32, BOOLtoI32MaxToNonMax) {
    const auto value = std::numeric_limits<fromType>::max();
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

class SaturateCastTestsU8ToI32 : public ov::test::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::U8>::value_type;
    using toType = typename PrecisionTrait<Precision::I32>::value_type;
};

TEST_F(SaturateCastTestsU8ToI32, U8toI32FMaxToNonMax) {
    const auto value = std::numeric_limits<fromType>::max();
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

class SaturateCastTestsU16ToI32 : public ov::test::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::U8>::value_type;
    using toType = typename PrecisionTrait<Precision::I32>::value_type;
};

TEST_F(SaturateCastTestsU16ToI32, U16toI32FMaxToNonMax) {
    const auto value = std::numeric_limits<fromType>::max();
    EXPECT_EQ(PrecisionUtils::saturate_cast<toType>(value), cv::saturate_cast<toType>(value));
}

#endif  // USE_OPENCV
