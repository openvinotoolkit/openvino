// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/strided_slice.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::StridedSliceLayerTest;
using ov::test::StridedSliceSpecificParams;
using ov::test::StridedSliceParams;


std::vector<StridedSliceSpecificParams> ss_only_test_cases_fp32 = {
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 128, 1 }})),
                                { 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 },
                                { 0, 1, 1 }, { 1, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 128, 1 }})),
                                { 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1},
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 1, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, -1, 0 }, { 0, 0, 0 }, { 1, 1, 1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, 9, 0 }, { 0, 11, 0 }, { 1, 1, 1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, 1, 0 }, { 0, -1, 0 }, { 1, 1, 1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, 9, 0 }, { 0, 7, 0 }, { -1, -1, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, 7, 0 }, { 0, 9, 0 }, { -1, 1, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, 4, 0 }, { 0, 9, 0 }, { -1, 2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, 4, 0 }, { 0, 10, 0 }, { -1, 2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, 9, 0 }, { 0, 4, 0 }, { -1, -2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, 10, 0 }, { 0, 4, 0 }, { -1, -2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, 11, 0 }, { 0, 0, 0 }, { -1, -2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, -6, 0 }, { 0, -8, 0 }, { -1, -2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, 6, 0 }, { 0, 4, 0 }, { -1, -2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100 }})),
                                { 0, -8, 0 }, { 0, -4, 0 }, { -1, 2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 12, 100, 1, 1 }})),
                                { 0, -1, 0, 0 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 },
                                { 1, 0, 1, 0 }, { 1, 0, 1, 0 }, { }, { 0, 1, 0, 1 }, {} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 2, 2 }})),
                                { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 2, 2 }})),
                                { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {0, 0, 0, 0}, {1, 1, 1, 1}, {}, {}, {} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 2, 128, 2 }})),
                                { 0, 0, 0, 1 }, { 0, 1, 0, 2 }, { 1, 1, 1, 1 },
                                {1, 0, 1, 0}, {1, 0, 1, 0}, {0, 0, 0, 0}, {0, 1, 0, 1}, {0, 0, 0, 0} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 2, 2 }})),
                                { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {0, 0, 0, 0}, {0, 0, 0, 0}, {}, {}, {} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 4, 3 }})),
                                { 0, 0, 0, 0 }, { 2, 2, 4, 3 }, { 1, 1, 2, 1 },
                                {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 4, 2 }})),
                                { 1, 0, 0, 1 }, { 2, 2, 4, 2 }, { 1, 1, 2, 1 },
                                {0, 1, 1, 0}, {1, 1, 0, 0}, {}, {}, {} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 2, 4, 2 }})),
                                { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                                {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 4, 2 }})),
                                { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                                {0, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 3, 4, 5, 6 }})),
                                { 0, 1, 0, 0, 0 }, { 2, 3, 4, 5, 6 }, { 1, 1, 1, 1, 1 },
                                {1, 0, 1, 1, 1}, {1, 0, 1, 1, 1}, {}, {0, 1, 0, 0, 0}, {} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 5, 30, 30, 30 }})),
                                { 0, 0, 0, 0, 0 }, { 0, 0, 29, 29, 29 }, { 1, 1, 1, 1, 1 },
                                {1, 1, 1, 1, 1}, {1, 1, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 10, 10 }})),
                                { 0, 0 }, { 1000, 2 }, { 1, 1 },
                                { 0, 1 }, { 0, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 10, 10 }})),
                                { -1000, 0 }, { 1000, 2 }, { 1, 1 },
                                { 0, 1 }, { 0, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 10, 10 }})),
                                { 1000, 1 }, { -1000, 2 }, { -1, 1 },
                                { 0, 1 }, { 0, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 10, 10 }})),
                                { -1, 1 }, { -1000, 2 }, { -1, 1 },
                                { 0, 1 }, { 0, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 10, 10 }})),
                                { -1, 1 }, { 0, 2 }, { -1, 1 },
                                { 0, 1 }, { 0, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 10, 10 }})),
                                { -4, 1 }, { -8, 0 }, { -1, 1 },
                                { 0, 1 }, { 0, 1 }, { 0, 0 }, { 0, 0 }, { 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 5, 5, 5, 5 }})),
                                { -1, 0, -1, 0 }, { -50, 0, -60, 0 }, { -1, 1, -1, 1 },
                                { 0, 0, 0, 0 }, { 0, 1, 0, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 4, 1 }})),
                                { 0, 0, 0, 0 }, { 2, 2, 4, 1 }, {  1, 1, 1, 1 },
                                { 0 }, { 0 }, { 1 }, { 0 }, {0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 128, 1, 1024 }})),
                                { -1, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 },
                                { 0, 1, 1 }, { 0, 1, 1 }, { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 10, 10 }})),
                                { -4, 1 }, { -8, 0 }, { -1, 1 },
                                { 0, 1 }, { 0, 1 }, { 0, 0 }, { 0, 0 }, { 0, 1 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 4, 1 }})),
                                { 0, 0 }, { 2, 2 }, {  -1, 1 },
                                { 1, 0 }, { 1, 0 }, { 0, 0 }, { 0, 0 }, { 0, 1 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 4, 1 }})),
                                { 0, 0 }, { 4, 1 }, {  1, -1 },
                                { 0, 1 }, { 0, 1 }, { 0, 0 }, { 0, 0 }, { 1, 0 } },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 5, 30, 30, 30 }})),
                                { 0, 0, 0 }, { 0, 29, 29 }, { 1, 1, 1 },
                                {1, 1, 1}, {1, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 1, 0} },
};

std::vector<StridedSliceSpecificParams> ss_only_test_cases_i64 = {
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 2, 2 }})),
                                { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {1, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 2, 2, 2 }})),
                                { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {0, 0, 0, 0}, {0, 0, 0, 0},  {},  {},  {} },
};

INSTANTIATE_TEST_SUITE_P(
        smoke_CLDNN_FP32, StridedSliceLayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(ss_only_test_cases_fp32),
            ::testing::Values(ov::element::f32),
            ::testing::Values(ov::test::utils::DEVICE_GPU)),
        StridedSliceLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_CLDNN_I64, StridedSliceLayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(ss_only_test_cases_i64),
            ::testing::Values(ov::element::i64),
            ::testing::Values(ov::test::utils::DEVICE_GPU)),
        StridedSliceLayerTest::getTestCaseName);

}  // namespace
