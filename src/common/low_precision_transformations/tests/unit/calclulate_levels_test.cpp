// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/network_helper.hpp"

using LPT_CalculateLevelsTestTransformation = ::testing::Test;

namespace {

size_t calculateLevels(const float dataPrecisionMin,
                       const float dataPrecisionMax,
                       const float combinedIntervalLow,
                       const float combinedIntervalHigh,
                       const float minIntervalLow,
                       const float minIntervalHigh) {
    float dequantizationMul;
    float dequantizationSub;
    float updatedOutputLowValue;
    float updatedOutputHighValue;

    const auto levels = ov::pass::low_precision::NetworkHelper::calculateLevels(dataPrecisionMin,
                                                                                    dataPrecisionMax,
                                                                                    combinedIntervalLow,
                                                                                    combinedIntervalHigh,
                                                                                    minIntervalLow,
                                                                                    minIntervalHigh,
                                                                                    dequantizationMul,
                                                                                    dequantizationSub,
                                                                                    updatedOutputLowValue,
                                                                                    updatedOutputHighValue);

    return levels;
}

}  // namespace
TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_U8_256) {
    const auto levels =
        calculateLevels(0.f, ov::pass::low_precision::DataPrecision::getMaxValue(256ul), 0.f, 2.55f, 0.f, 2.55f);
    ASSERT_EQ(256ul, levels);
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_I8_256) {
    const auto levels = calculateLevels(0.f,
                                        ov::pass::low_precision::DataPrecision::getMaxValue(256ul),
                                        -1.28f,
                                        1.27f,
                                        -1.28f,
                                        1.27f);
    ASSERT_EQ(256ul, levels);
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_U8_128) {
    const auto levels = calculateLevels(0.f,
                                        ov::pass::low_precision::DataPrecision::getMaxValue(256ul),
                                        0.f,
                                        2.55f,
                                        0.f,
                                        2.55f / 2.f);
    ASSERT_EQ(129ul, levels);
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_I8_128) {
    const auto levels = calculateLevels(0.f,
                                        ov::pass::low_precision::DataPrecision::getMaxValue(256ul),
                                        -1.28f,
                                        1.27f,
                                        -1.28f / 2.f,
                                        1.27f / 2.f);
    ASSERT_EQ(129ul, levels);
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_0) {
    const auto levels =
        calculateLevels(0.f, ov::pass::low_precision::DataPrecision::getMaxValue(256ul), 0.f, 2.55f, 0.f, 0.f);
    ASSERT_EQ(1ul, levels);
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_3) {
    const auto levels =
        calculateLevels(0.f, ov::pass::low_precision::DataPrecision::getMaxValue(256ul), 0.f, 2.55f, 0.f, 0.0255f);
    ASSERT_EQ(4ul, levels);
}
