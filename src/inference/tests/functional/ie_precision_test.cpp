// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_common.h>

#include <ie_precision.hpp>
#include <string>

IE_SUPPRESS_DEPRECATED_START

using Precision = InferenceEngine::Precision;

using PrecisionTests = ::testing::Test;

TEST_F(PrecisionTests, ShowsCorrectPrecisionNames) {
    EXPECT_STREQ(Precision(Precision::I64).name(), "I64");
    EXPECT_STREQ(Precision(Precision::U64).name(), "U64");
    EXPECT_STREQ(Precision(Precision::BF16).name(), "BF16");
    EXPECT_STREQ(Precision(Precision::FP16).name(), "FP16");
    EXPECT_STREQ(Precision(Precision::FP32).name(), "FP32");
    EXPECT_STREQ(Precision(Precision::FP64).name(), "FP64");
    EXPECT_STREQ(Precision(Precision::I16).name(), "I16");
    EXPECT_STREQ(Precision(Precision::I32).name(), "I32");
    EXPECT_STREQ(Precision(Precision::U32).name(), "U32");
    EXPECT_STREQ(Precision(Precision::U16).name(), "U16");
    EXPECT_STREQ(Precision(Precision::I4).name(), "I4");
    EXPECT_STREQ(Precision(Precision::I8).name(), "I8");
    EXPECT_STREQ(Precision(Precision::Q78).name(), "Q78");
    EXPECT_STREQ(Precision(Precision::U4).name(), "U4");
    EXPECT_STREQ(Precision(Precision::U8).name(), "U8");
    EXPECT_STREQ(Precision(Precision::MIXED).name(), "MIXED");
    EXPECT_STREQ(Precision(Precision::UNSPECIFIED).name(), "UNSPECIFIED");
    EXPECT_STREQ(Precision(static_cast<Precision::ePrecision>(-3)).name(), "UNSPECIFIED");
    EXPECT_STREQ(Precision(1, "Custom Name").name(), "Custom Name");
}

TEST_F(PrecisionTests, sizeIsCorrect) {
    EXPECT_EQ(Precision(Precision::I64).size(), 8);
    EXPECT_EQ(Precision(Precision::U64).size(), 8);
    EXPECT_EQ(Precision(Precision::BF16).size(), 2);
    EXPECT_EQ(Precision(Precision::FP16).size(), 2);
    EXPECT_EQ(Precision(Precision::FP32).size(), 4);
    EXPECT_EQ(Precision(Precision::FP64).size(), 8);
    EXPECT_EQ(Precision(Precision::I32).size(), 4);
    EXPECT_EQ(Precision(Precision::U32).size(), 4);
    EXPECT_EQ(Precision(Precision::I16).size(), 2);
    EXPECT_EQ(Precision(Precision::U16).size(), 2);
    EXPECT_EQ(Precision(Precision::I4).size(), 1);
    EXPECT_EQ(Precision(Precision::I8).size(), 1);
    EXPECT_EQ(Precision(Precision::Q78).size(), 2);
    EXPECT_EQ(Precision(Precision::U8).size(), 1);
    EXPECT_EQ(Precision(Precision::U4).size(), 1);
    EXPECT_EQ(Precision(10 * 8).size(), 10);
    EXPECT_THROW(Precision(Precision::MIXED).size(), InferenceEngine::Exception);
    EXPECT_THROW(Precision(Precision::UNSPECIFIED).size(), InferenceEngine::Exception);
}

TEST_F(PrecisionTests, bitsSizeIsCorrect) {
    EXPECT_EQ(Precision(Precision::I64).bitsSize(), 64);
    EXPECT_EQ(Precision(Precision::U64).bitsSize(), 64);
    EXPECT_EQ(Precision(Precision::BF16).bitsSize(), 16);
    EXPECT_EQ(Precision(Precision::FP16).bitsSize(), 16);
    EXPECT_EQ(Precision(Precision::FP32).bitsSize(), 32);
    EXPECT_EQ(Precision(Precision::FP64).bitsSize(), 64);
    EXPECT_EQ(Precision(Precision::I32).bitsSize(), 32);
    EXPECT_EQ(Precision(Precision::U32).bitsSize(), 32);
    EXPECT_EQ(Precision(Precision::I16).bitsSize(), 16);
    EXPECT_EQ(Precision(Precision::U16).bitsSize(), 16);
    EXPECT_EQ(Precision(Precision::I4).bitsSize(), 4);
    EXPECT_EQ(Precision(Precision::I8).bitsSize(), 8);
    EXPECT_EQ(Precision(Precision::Q78).bitsSize(), 16);
    EXPECT_EQ(Precision(Precision::U8).bitsSize(), 8);
    EXPECT_EQ(Precision(Precision::U4).bitsSize(), 4);
    EXPECT_EQ(Precision(10 * 8).bitsSize(), 80);
    EXPECT_THROW(Precision(Precision::MIXED).bitsSize(), InferenceEngine::Exception);
    EXPECT_THROW(Precision(Precision::UNSPECIFIED).bitsSize(), InferenceEngine::Exception);
}

TEST_F(PrecisionTests, is_float) {
    EXPECT_TRUE(Precision(Precision::BF16).is_float());
    EXPECT_TRUE(Precision(Precision::FP16).is_float());
    EXPECT_TRUE(Precision(Precision::FP32).is_float());
    EXPECT_TRUE(Precision(Precision::FP64).is_float());
    EXPECT_FALSE(Precision(Precision::I64).is_float());
    EXPECT_FALSE(Precision(Precision::U64).is_float());
    EXPECT_FALSE(Precision(Precision::I32).is_float());
    EXPECT_FALSE(Precision(Precision::U32).is_float());
    EXPECT_FALSE(Precision(Precision::I16).is_float());
    EXPECT_FALSE(Precision(Precision::U16).is_float());
    EXPECT_FALSE(Precision(Precision::I8).is_float());
    EXPECT_FALSE(Precision(Precision::I4).is_float());
    EXPECT_FALSE(Precision(Precision::Q78).is_float());
    EXPECT_FALSE(Precision(Precision::U4).is_float());
    EXPECT_FALSE(Precision(Precision::U8).is_float());
    EXPECT_FALSE(Precision(Precision::MIXED).is_float());
    EXPECT_FALSE(Precision(10).is_float());
    EXPECT_FALSE(Precision(static_cast<Precision::ePrecision>(-3)).is_float());
    EXPECT_FALSE(Precision(Precision::UNSPECIFIED).is_float());
}

TEST_F(PrecisionTests, constructFromSTR) {
    EXPECT_EQ(Precision(Precision::I64), Precision::FromStr("I64"));
    EXPECT_EQ(Precision(Precision::U64), Precision::FromStr("U64"));
    EXPECT_EQ(Precision(Precision::BF16), Precision::FromStr("BF16"));
    EXPECT_EQ(Precision(Precision::FP16), Precision::FromStr("FP16"));
    EXPECT_EQ(Precision(Precision::FP32), Precision::FromStr("FP32"));
    EXPECT_EQ(Precision(Precision::FP64), Precision::FromStr("FP64"));
    EXPECT_EQ(Precision(Precision::I32), Precision::FromStr("I32"));
    EXPECT_EQ(Precision(Precision::U32), Precision::FromStr("U32"));
    EXPECT_EQ(Precision(Precision::I16), Precision::FromStr("I16"));
    EXPECT_EQ(Precision(Precision::U16), Precision::FromStr("U16"));
    EXPECT_EQ(Precision(Precision::I4), Precision::FromStr("I4"));
    EXPECT_EQ(Precision(Precision::I8), Precision::FromStr("I8"));
    EXPECT_EQ(Precision(Precision::Q78), Precision::FromStr("Q78"));
    EXPECT_EQ(Precision(Precision::U4), Precision::FromStr("U4"));
    EXPECT_EQ(Precision(Precision::U8), Precision::FromStr("U8"));
    EXPECT_EQ(Precision(Precision::MIXED), Precision::FromStr("MIXED"));
    EXPECT_EQ(Precision(static_cast<Precision::ePrecision>(-3)), Precision::FromStr("UNSPECIFIED"));
    EXPECT_EQ(Precision(Precision::UNSPECIFIED), Precision::FromStr("UNSPECIFIED"));
}

TEST_F(PrecisionTests, canCompareCustomPrecisions) {
    Precision p(12);
    Precision p1(12, "XXX");
    EXPECT_FALSE(p == p1);

    std::string d;
    d.push_back('X');
    d.push_back('X');
    d.push_back('X');
    Precision p2(12, d.c_str());
    EXPECT_TRUE(p2 == p1);

    Precision p3(13, "XXX");
    EXPECT_FALSE(p3 == p1);

    Precision p4(13);
    EXPECT_FALSE(p4 == p);

    Precision p5(12);
    EXPECT_TRUE(p5 == p);
}

TEST_F(PrecisionTests, canUseInIfs) {
    Precision p;
    EXPECT_TRUE(!p);
    p = Precision::FP32;
    EXPECT_FALSE(!p);
    EXPECT_TRUE(p);
    p = Precision(static_cast<Precision::ePrecision>(-3));
    EXPECT_TRUE(!p);
}

TEST_F(PrecisionTests, canCreateFromStruct) {
    struct X {
        int a;
        int b;
    };
    auto precision = Precision::fromType<X>();
    EXPECT_EQ(precision.size(), sizeof(X));
}

TEST_F(PrecisionTests, canCreateMoreThan255bitsPrecisions) {
    struct Y {
        uint8_t a[257];
    };

    EXPECT_NO_THROW(Precision::fromType<Y>());
    EXPECT_EQ(Precision::fromType<Y>().size(), 257);
}
