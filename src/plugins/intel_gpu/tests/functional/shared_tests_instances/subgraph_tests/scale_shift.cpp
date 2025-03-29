// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/scaleshift.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ScaleShiftLayerTest;

std::vector<std::vector<ov::Shape>> inShapes = {
        {{100}},
        {{100}, {100}},
        {{1, 8}},
        {{2, 16}},
        {{3, 32}},
        {{4, 64}},
        {{4, 64}, {64}},
        {{5, 128}},
        {{6, 256}},
        {{7, 512}},
        {{8, 1024}}
};

std::vector<std::vector<float>> Scales = {
        {2.0f},
        {3.0f},
        {-1.0f},
        {-2.0f},
        {-3.0f}
};

std::vector<std::vector<float>> Shifts = {
        {1.0f},
        {2.0f},
        {3.0f},
        {-1.0f},
        {-2.0f},
        {-3.0f}
};

std::vector<ov::element::Type> types = {ov::element::f32,
                                        ov::element::f16,
};


INSTANTIATE_TEST_SUITE_P(smoke_ScaleShift, ScaleShiftLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(types),
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(Scales),
                                ::testing::ValuesIn(Shifts)),
                        ScaleShiftLayerTest::getTestCaseName);
}  // namespace
