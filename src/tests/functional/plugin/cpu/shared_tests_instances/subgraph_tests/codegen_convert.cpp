
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/codegen_convert.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32
};

const std::vector<std::pair<ov::element::Type, ov::element::Type>> convertPrecisions = {
    {ov::element::u8, ov::element::f32},
    {ov::element::i8, ov::element::f32},
    // common transformations fail on TypeRelaxed type
    //{ov::element::f32, ov::element::u8},
    //{ov::element::f32, ov::element::i8},
};

INSTANTIATE_TEST_SUITE_P(CodeGeneration, CodegenConvert,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::SizeVector({1, 3, 1024 * 4, 1024 * 4})),
            ::testing::ValuesIn(convertPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        CodegenConvert::getTestCaseName);
}  // namespace
