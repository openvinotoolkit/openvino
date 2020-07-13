// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/test_common.hpp"

#include "saturated_cast.hpp"

#include "ie_precision.hpp"

using namespace InferenceEngine;

class SaturatedCastTestsI64ToI32 : public CommonTestUtils::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::I64>::value_type;
    using toType = typename PrecisionTrait<Precision::I32>::value_type;
};

TEST_F(SaturatedCastTestsI64ToI32, I64ToI32NonNarrowingPositive) {
    EXPECT_EQ(saturated_cast<toType>(fromType{42}), toType{42});
}

TEST_F(SaturatedCastTestsI64ToI32, I64ToI32NonNarrowingNegative) {
    EXPECT_EQ(saturated_cast<toType>(fromType{-42}), toType{-42});
}

TEST_F(SaturatedCastTestsI64ToI32, I64ToI32NarrowingMaxToMax) {
    EXPECT_EQ(saturated_cast<toType>(std::numeric_limits<fromType>::max()), std::numeric_limits<toType>::max());
}

TEST_F(SaturatedCastTestsI64ToI32, I64ToI32NarrowingNonMaxToMax) {
    EXPECT_EQ(saturated_cast<toType>(std::numeric_limits<fromType>::max() - 1), std::numeric_limits<toType>::max());
}

TEST_F(SaturatedCastTestsI64ToI32, I64ToI32NarrowingMinToMin) {
    EXPECT_EQ(saturated_cast<toType>(std::numeric_limits<fromType>::min()), std::numeric_limits<toType>::min());
}

TEST_F(SaturatedCastTestsI64ToI32, I64ToI32NarrowingNonMinToMin) {
    EXPECT_EQ(saturated_cast<toType>(std::numeric_limits<fromType>::min() + 1), std::numeric_limits<toType>::min());
}

class SaturatedCastTestsU64ToI32 : public CommonTestUtils::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::U64>::value_type;
    using toType = typename PrecisionTrait<Precision::I32>::value_type;
};

TEST_F(SaturatedCastTestsU64ToI32, U64ToI32NonNarrowing) {
    EXPECT_EQ(saturated_cast<toType>(fromType{42}), toType{42});
}

TEST_F(SaturatedCastTestsU64ToI32, U64ToI32NarrowingMaxToMax) {
    EXPECT_EQ(saturated_cast<toType>(std::numeric_limits<fromType>::max()), std::numeric_limits<toType>::max());
}

TEST_F(SaturatedCastTestsU64ToI32, U64ToI32NarrowingNonMaxToMax) {
    EXPECT_EQ(saturated_cast<toType>(std::numeric_limits<fromType>::max() - 1), std::numeric_limits<toType>::max());
}

class SaturatedCastTestsBoolToU8 : public CommonTestUtils::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::BOOL>::value_type;
    using toType = typename PrecisionTrait<Precision::U8>::value_type;
};

TEST_F(SaturatedCastTestsBoolToU8, BOOLtoU8MaxToNonMax) {
    EXPECT_EQ(saturated_cast<toType>(std::numeric_limits<fromType>::max()), toType{std::numeric_limits<fromType>::max()});
}

class SaturatedCastTestsBoolToI32 : public CommonTestUtils::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::BOOL>::value_type;
    using toType = typename PrecisionTrait<Precision::I32>::value_type;
};

TEST_F(SaturatedCastTestsBoolToI32, BOOLtoI32MaxToNonMax) {
    EXPECT_EQ(saturated_cast<toType>(std::numeric_limits<fromType>::max()), toType{std::numeric_limits<fromType>::max()});
}

class SaturatedCastTestsU8ToI32 : public CommonTestUtils::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::U8>::value_type;
    using toType = typename PrecisionTrait<Precision::I32>::value_type;
};

TEST_F(SaturatedCastTestsU8ToI32, U8toI32FMaxToNonMax) {
    EXPECT_EQ(saturated_cast<toType>(std::numeric_limits<fromType>::max()), toType{std::numeric_limits<fromType>::max()});
}

class SaturatedCastTestsU16ToI32 : public CommonTestUtils::TestsCommon {
public:
    using fromType = typename PrecisionTrait<Precision::U8>::value_type;
    using toType = typename PrecisionTrait<Precision::I32>::value_type;
};

TEST_F(SaturatedCastTestsU16ToI32, U16toI32FMaxToNonMax) {
    EXPECT_EQ(saturated_cast<toType>(std::numeric_limits<fromType>::max()), toType{std::numeric_limits<fromType>::max()});
}
