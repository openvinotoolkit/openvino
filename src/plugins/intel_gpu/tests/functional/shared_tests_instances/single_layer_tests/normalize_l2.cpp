// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/normalize_l2.hpp"

namespace {
using ov::test::NormalizeL2LayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<std::vector<int64_t>> axes = {
        {},
        {1},
};
const std::vector<float> eps = {1e-7f, 1e-6f, 1e-5f, 1e-4f};

const std::vector<ov::op::EpsMode> epsMode = {
        ov::op::EpsMode::ADD,
        ov::op::EpsMode::MAX,
};

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2,
                         NormalizeL2LayerTest,
                         testing::Combine(testing::ValuesIn(axes),
                                          testing::ValuesIn(eps),
                                          testing::ValuesIn(epsMode),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                std::vector<ov::Shape>{{1, 3, 10, 5}})),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         NormalizeL2LayerTest::getTestCaseName);
}  // namespace
