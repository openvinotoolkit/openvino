// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "low_precision_transformations/mat_mul_transformation.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
        ov::element::f32
};

std::vector<MatMulTransformationTestValues> testValues = {
    // supported on arm only
    {
        { 1, 1, 12, 2 },
        { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
        { 1, 1, 2, 12 },
        { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
        "matMul",
        "u8",
        false
    },
    // supported on arm only: LPT was updated
    {
        { 1, 1, 12, 2 },
        { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
        { 1, 1, 2, 12 },
        { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
        "matMul",
        "u8",
        true
    },
//    // supported on arm only: should update LPT and uncomment
//    {
//        { 1, 1, 12, 2 },
//        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
//        { 1, 1, 2, 12 },
//        { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
//        "matMul",
//        "u8",
//        false
//    },

//    {
//        { 1, 4, 12, 2 },
//        { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
//        { 1, 4, 2, 12 },
//        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
//        "matMul_original",
//        "u8"
//    },
//    {
//        { 8, 4, 12, 2 },
//        { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
//        { 8, 4, 2, 12 },
//        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
//        "matMul_original",
//        "u8"
//    },

    {
        { 1, 1, 12, 8 },
        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
        { 1, 1, 8, 6 },
        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
        "matMul",
        "i8",
        false
    },
    {
        { 1, 1, 12, 2 },
        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
        { 1, 1, 2, 12 },
        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
        "matMul",
        "i8",
        true
    },

//    {
//        { 1, 1, 1, 4, 12, 2 },
//        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
//        { 1, 1, 1, 4, 2, 12 },
//        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
//        "matMul_original",
//        "i8"
//    },
    {
        { 12 },
        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
        { 12 },
        { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
        "matMul_original/MM",
        "i8",
        false,
    },
    {
        { 12 },
        { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
        { 12 },
        { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
        "matMul_original/MM",
        "u8",
        false,
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, MatMulTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(ov::PartialShape({ 1, 384, 1024 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(testValues)),
    MatMulTransformation::getTestCaseName);
}  // namespace
