// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/extract_image_patches.hpp"


namespace {
using ov::test::ExtractImagePatchesTest;
using ov::op::PadType;

const std::vector<std::vector<ov::Shape>> inDataShape = {
    {{1, 1, 10, 10}},
    {{1, 3, 10, 10}}
};
const std::vector<std::vector<size_t>> kernels = {
    {2, 2},
    {3, 3},
    {4, 4},
    {1, 3},
    {4, 2}
};
const std::vector<std::vector<size_t>> strides = {
    {3, 3},
    {5, 5},
    {9, 9},
    {1, 3},
    {6, 2}
};
const std::vector<std::vector<size_t>> rates = {
    {1, 1},
    {1, 2},
    {2, 1},
    {2, 2}
};
const std::vector<PadType> autoPads = {
    PadType::VALID,
    PadType::SAME_UPPER,
    PadType::SAME_LOWER
};
const std::vector<ov::element::Type> netPrecisions = {
    //ov::element::i8,
    ov::element::u8,
    ov::element::i16,
    ov::element::i32,
    ov::element::f32
};

INSTANTIATE_TEST_SUITE_P(smoke_layers_GPU1, ExtractImagePatchesTest,
        ::testing::Combine(
            ::testing::Values(ov::test::static_shapes_to_test_representation(inDataShape)[0]),
            ::testing::Values(kernels[0]),
            ::testing::Values(strides[0]),
            ::testing::ValuesIn(rates),
            ::testing::ValuesIn(autoPads),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(ov::test::utils::DEVICE_GPU)),
        ExtractImagePatchesTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_layers_GPU2, ExtractImagePatchesTest,
        ::testing::Combine(
            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inDataShape)),
            ::testing::ValuesIn(kernels),
            ::testing::ValuesIn(strides),
            ::testing::Values(rates[0]),
            ::testing::Values(autoPads[0]),
            ::testing::Values(netPrecisions[0]),
            ::testing::Values(ov::test::utils::DEVICE_GPU)),
        ExtractImagePatchesTest::getTestCaseName);

}  // namespace
