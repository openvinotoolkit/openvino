// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "low_precision/reshape.hpp"

using LPT_ReshapeTransformation = ::testing::Test;

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_2D_perTensor) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({}),
        ov::Shape({}),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 1, 60 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_2D_perTensor2) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1 }),
        ov::Shape({ 1 }),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 1, 60 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_2D_perChannels) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3 }),
        ov::Shape({ 1, 3 }),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 1, 60 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_2D_perChannels2) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3, 1, 1 }),
        ov::Shape({ 1, 3, 1, 1 }),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 1, 60 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_2D_perChannels3) {
    ASSERT_FALSE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3, 1, 1 }),
        ov::Shape({ 1, 3, 1, 1 }),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 12, 5 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_2D_spatial) {
    ASSERT_FALSE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3, 4, 1 }),
        ov::Shape({ 1, 3, 4, 1 }),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 1, 60 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_3D_perTensor) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({}),
        ov::Shape({}),
        ov::Shape({1, 3, 4, 5}),
        ov::Shape({1, 3, 20})));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_3D_perTensor2) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1 }),
        ov::Shape({ 1 }),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 1, 3, 20 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_3D_perChannels) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3 }),
        ov::Shape({ 1, 3 }),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 1, 3, 20 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_3D_perChannels2) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3, 1, 1 }),
        ov::Shape({ 1, 3, 1, 1 }),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 1, 3, 20 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_3D_perSpacial1) {
    ASSERT_FALSE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3, 4, 1 }),
        ov::Shape({ 1, 3, 4, 1 }),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 1, 3, 20 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_3D_perSpacial2) {
    ASSERT_FALSE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3, 1, 4 }),
        ov::Shape({ 1, 3, 1, 4 }),
        ov::Shape({ 1, 3, 4, 5 }),
        ov::Shape({ 1, 3, 20 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_3D_to_4D_perTensor) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({}),
        ov::Shape({}),
        ov::Shape({ 1, 3, 20 }),
        ov::Shape({ 1, 3, 4, 5 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_3D_to_4D_perTensor2) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1 }),
        ov::Shape({ 1 }),
        ov::Shape({ 1, 3, 20 }),
        ov::Shape({ 1, 3, 4, 5 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_3D_to_4D_perChannels) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3 }),
        ov::Shape({ 1, 3 }),
        ov::Shape({ 1, 3, 20 }),
        ov::Shape({ 1, 3, 4, 5 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_3D_to_4D_perChannels2) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3, 1, 1 }),
        ov::Shape({ 1, 3, 1, 1 }),
        ov::Shape({ 1, 3, 20 }),
        ov::Shape({ 1, 3, 4, 5 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_3D_to_4D_perSpacial) {
    ASSERT_FALSE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 3, 20 }),
        ov::Shape({ 1, 3, 20 }),
        ov::Shape({ 1, 3, 20 }),
        ov::Shape({ 1, 3, 4, 5 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_4D_to_2D_perSpacial_TRUE) {
    ASSERT_TRUE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 256, 1, 1 }),
        ov::Shape({ 1, 256, 1, 1 }),
        ov::Shape({ 1, 256, 6, 6 }),
        ov::Shape({ 1, 9216 })));
}

TEST(LPT_ReshapeTransformation, canBeTransformed_5D_to_5D_perBatch) {
    ASSERT_FALSE(ov::pass::low_precision::ReshapeTransformation::canBeTransformed(
        ov::Shape({ 1, 16, 1, 1, 1 }),
        ov::Shape({ 1, 16, 1, 1, 1 }),
        ov::Shape({ 1, 16, 128, 128, 128 }),
        ov::Shape({ 8, 2, 128, 128, 128 })));
}
