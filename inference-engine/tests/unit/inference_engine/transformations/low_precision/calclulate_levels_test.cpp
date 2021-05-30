// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <gtest/gtest.h>
#include "low_precision/network_helper.hpp"

using LPT_CalculateLevelsTestTransformation = ::testing::Test;

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_U8_256) {
    ASSERT_EQ(256ul, ngraph::pass::low_precision::NetworkHelper::calculateLevels(256ul, 0.f, 2.55f, 0.f, 2.55f));
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_I8_256) {
    ASSERT_EQ(256ul, ngraph::pass::low_precision::NetworkHelper::calculateLevels(256ul, -1.28f, 1.27f, -1.28f, 1.27f));
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_U8_128) {
    ASSERT_EQ(128ul, ngraph::pass::low_precision::NetworkHelper::calculateLevels(256ul, 0.f, 2.55f, 0.f, 2.55f / 2.f));
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_U8I8_128) {
    ASSERT_EQ(128ul, ngraph::pass::low_precision::NetworkHelper::calculateLevels(256ul, 0.f, 2.55f, -1.28f / 2.f, 1.27f / 2.f));
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_I8_128) {
    ASSERT_EQ(128ul, ngraph::pass::low_precision::NetworkHelper::calculateLevels(256ul, -1.28f, 1.27f, -1.28f / 2.f, 1.27f / 2.f));
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_I8U8_128) {
    ASSERT_EQ(128ul, ngraph::pass::low_precision::NetworkHelper::calculateLevels(256ul, -1.28f, 1.27f, 0.f, 2.55f / 2.f));
}

TEST(LPT_CalculateLevelsTestTransformation, calculateLevels_0) {
    auto levels = ngraph::pass::low_precision::NetworkHelper::calculateLevels(0ul, 0.f, 2.55f, 0.f, 0.f);
    ASSERT_EQ(0ul, levels);
}
