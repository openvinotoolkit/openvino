// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_utils.h"

#include <gtest/gtest.h>

#include <limits>

using namespace InferenceEngine;

using PrecisionUtilsTests = ::testing::Test;

static constexpr ie_fp16 positiveInf = 0x7C00;
static constexpr ie_fp16 negativeInf = 0xFC00;
static constexpr ie_fp16 largestNumber = 0x7BFF;
static constexpr ie_fp16 lowestNumber = 0xFBFF;

TEST_F(PrecisionUtilsTests, FP32ToFP16PositiveInfinity) {
    const auto fp16ConvertedInf = InferenceEngine::PrecisionUtils::f32tof16(std::numeric_limits<float>::infinity());
    ASSERT_EQ(fp16ConvertedInf, positiveInf);
}

TEST_F(PrecisionUtilsTests, FP32ToFP16NegativeInfinity) {
    const auto fp16ConvertedInf = InferenceEngine::PrecisionUtils::f32tof16(-1 * std::numeric_limits<float>::infinity());
    ASSERT_EQ(fp16ConvertedInf, negativeInf);
}

TEST_F(PrecisionUtilsTests, FP16ToFP32PositiveInfinity) {
    const auto fp32ConvertedInf = InferenceEngine::PrecisionUtils::f16tof32(positiveInf);
    ASSERT_EQ(fp32ConvertedInf, std::numeric_limits<float>::infinity());
}

TEST_F(PrecisionUtilsTests, FP16ToFP32NegativeInfinity) {
    const auto fp32ConvertedInf = InferenceEngine::PrecisionUtils::f16tof32(negativeInf);
    ASSERT_EQ(fp32ConvertedInf, -1 * std::numeric_limits<float>::infinity());
}

TEST_F(PrecisionUtilsTests, FP32ToFP16MaximumValue) {
    const auto fp16ConvertedMaxValue = InferenceEngine::PrecisionUtils::f32tof16(std::numeric_limits<float>::max());
    ASSERT_EQ(fp16ConvertedMaxValue, largestNumber);
}

TEST_F(PrecisionUtilsTests, FP32ToFP16LowestValue) {
    const auto fp16ConvertedLowestValue = InferenceEngine::PrecisionUtils::f32tof16(std::numeric_limits<float>::lowest());
    ASSERT_EQ(fp16ConvertedLowestValue, lowestNumber);
}
