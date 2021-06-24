// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <gtest/gtest.h>

#include <ie_precision.hpp>
#include <ie_common.h>

using Precision = InferenceEngine::Precision;

using PrecisionTests = ::testing::Test;

TEST_F(PrecisionTests, ShowsCorrectPrecisionNames) {
    ASSERT_STREQ(Precision(Precision::I64).name(), "I64");
    ASSERT_STREQ(Precision(Precision::U64).name(), "U64");
    ASSERT_STREQ(Precision(Precision::BF16).name(), "BF16");
    ASSERT_STREQ(Precision(Precision::FP16).name(), "FP16");
    ASSERT_STREQ(Precision(Precision::FP32).name(), "FP32");
    ASSERT_STREQ(Precision(Precision::FP64).name(), "FP64");
    ASSERT_STREQ(Precision(Precision::I16).name(), "I16");
    ASSERT_STREQ(Precision(Precision::I32).name(), "I32");
    ASSERT_STREQ(Precision(Precision::U32).name(), "U32");
    ASSERT_STREQ(Precision(Precision::U16).name(), "U16");
    ASSERT_STREQ(Precision(Precision::I4).name(), "I4");
    ASSERT_STREQ(Precision(Precision::I8).name(), "I8");
    ASSERT_STREQ(Precision(Precision::Q78).name(), "Q78");
    ASSERT_STREQ(Precision(Precision::U4).name(), "U4");
    ASSERT_STREQ(Precision(Precision::U8).name(), "U8");
    ASSERT_STREQ(Precision(Precision::MIXED).name(), "MIXED");
    ASSERT_STREQ(Precision(Precision::UNSPECIFIED).name(), "UNSPECIFIED");
    ASSERT_STREQ(Precision(static_cast<Precision::ePrecision >(-3)).name(), "UNSPECIFIED");
    ASSERT_STREQ(Precision(1, "Custom Name").name(), "Custom Name");
}

TEST_F(PrecisionTests, sizeIsCorrect) {
    ASSERT_EQ(Precision(Precision::I64).size(), 8);
    ASSERT_EQ(Precision(Precision::U64).size(), 8);
    ASSERT_EQ(Precision(Precision::BF16).size(), 2);
    ASSERT_EQ(Precision(Precision::FP16).size(), 2);
    ASSERT_EQ(Precision(Precision::FP32).size(), 4);
    ASSERT_EQ(Precision(Precision::FP64).size(), 8);
    ASSERT_EQ(Precision(Precision::I32).size(), 4);
    ASSERT_EQ(Precision(Precision::U32).size(), 4);
    ASSERT_EQ(Precision(Precision::I16).size(), 2);
    ASSERT_EQ(Precision(Precision::U16).size(), 2);
    ASSERT_EQ(Precision(Precision::I4).size(), 1);
    ASSERT_EQ(Precision(Precision::I8).size(), 1);
    ASSERT_EQ(Precision(Precision::Q78).size(), 2);
    ASSERT_EQ(Precision(Precision::U8).size(), 1);
    ASSERT_EQ(Precision(Precision::U4).size(), 1);
    ASSERT_EQ(Precision(10 * 8).size(), 10);
    ASSERT_ANY_THROW(Precision(Precision::MIXED).size());
    ASSERT_ANY_THROW(Precision(Precision::UNSPECIFIED).size());
}

TEST_F(PrecisionTests, is_float) {
    ASSERT_TRUE(Precision(Precision::BF16).is_float());
    ASSERT_TRUE(Precision(Precision::FP16).is_float());
    ASSERT_TRUE(Precision(Precision::FP32).is_float());
    ASSERT_TRUE(Precision(Precision::FP64).is_float());
    ASSERT_FALSE(Precision(Precision::I64).is_float());
    ASSERT_FALSE(Precision(Precision::U64).is_float());
    ASSERT_FALSE(Precision(Precision::I32).is_float());
    ASSERT_FALSE(Precision(Precision::U32).is_float());
    ASSERT_FALSE(Precision(Precision::I16).is_float());
    ASSERT_FALSE(Precision(Precision::U16).is_float());
    ASSERT_FALSE(Precision(Precision::I8).is_float());
    ASSERT_FALSE(Precision(Precision::I4).is_float());
    ASSERT_FALSE(Precision(Precision::Q78).is_float());
    ASSERT_FALSE(Precision(Precision::U4).is_float());
    ASSERT_FALSE(Precision(Precision::U8).is_float());
    ASSERT_FALSE(Precision(Precision::MIXED).is_float());
    ASSERT_FALSE(Precision(10).is_float());
    ASSERT_FALSE(Precision(static_cast<Precision::ePrecision >(-3)).is_float());
    ASSERT_FALSE(Precision(Precision::UNSPECIFIED).is_float());
}

TEST_F(PrecisionTests, constructFromSTR) {
    ASSERT_EQ(Precision(Precision::I64), Precision::FromStr("I64"));
    ASSERT_EQ(Precision(Precision::U64), Precision::FromStr("U64"));
    ASSERT_EQ(Precision(Precision::BF16), Precision::FromStr("BF16"));
    ASSERT_EQ(Precision(Precision::FP16), Precision::FromStr("FP16"));
    ASSERT_EQ(Precision(Precision::FP32), Precision::FromStr("FP32"));
    ASSERT_EQ(Precision(Precision::FP64), Precision::FromStr("FP64"));
    ASSERT_EQ(Precision(Precision::I32), Precision::FromStr("I32"));
    ASSERT_EQ(Precision(Precision::U32), Precision::FromStr("U32"));
    ASSERT_EQ(Precision(Precision::I16), Precision::FromStr("I16"));
    ASSERT_EQ(Precision(Precision::U16), Precision::FromStr("U16"));
    ASSERT_EQ(Precision(Precision::I4), Precision::FromStr("I4"));
    ASSERT_EQ(Precision(Precision::I8), Precision::FromStr("I8"));
    ASSERT_EQ(Precision(Precision::Q78), Precision::FromStr("Q78"));
    ASSERT_EQ(Precision(Precision::U4), Precision::FromStr("U4"));
    ASSERT_EQ(Precision(Precision::U8), Precision::FromStr("U8"));
    ASSERT_EQ(Precision(Precision::MIXED), Precision::FromStr("MIXED"));
    ASSERT_EQ(Precision(static_cast<Precision::ePrecision >(-3)), Precision::FromStr("UNSPECIFIED"));
    ASSERT_EQ(Precision(Precision::UNSPECIFIED), Precision::FromStr("UNSPECIFIED"));
}

TEST_F(PrecisionTests, canCompareCustomPrecisions) {
    Precision p(12);
    Precision p1(12, "XXX");
    ASSERT_FALSE(p == p1);

    std::string d;
    d.push_back('X');
    d.push_back('X');
    d.push_back('X');
    Precision p2(12, d.c_str());
    ASSERT_TRUE(p2 == p1);

    Precision p3(13, "XXX");
    ASSERT_FALSE(p3 == p1);

    Precision p4(13);
    ASSERT_FALSE(p4 == p);

    Precision p5(12);
    ASSERT_TRUE(p5 == p);
}


TEST_F(PrecisionTests, canUseInIfs) {
    Precision p;
    ASSERT_TRUE(!p);
    p = Precision::FP32;
    ASSERT_FALSE(!p);
    ASSERT_TRUE(p);
    p = Precision(static_cast<Precision::ePrecision >(-3));
    ASSERT_TRUE(!p);
}

TEST_F(PrecisionTests, canCreateFromStruct) {
    struct X {
        int a;
        int b;
    };
    auto precision = Precision::fromType<X>();
    ASSERT_EQ(precision.size(), sizeof(X));
}

TEST_F(PrecisionTests, canCreateMoreThan255bitsPrecisions) {
    struct Y {
        uint8_t a[257];
    };

    ASSERT_NO_THROW(Precision::fromType<Y>());
    ASSERT_EQ(Precision::fromType<Y>().size(), 257);
}
