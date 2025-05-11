// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/fake_quantize.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::FakeQuantizeLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<std::vector<ov::Shape>> inputShapes = {{{1, 1, 1, 1}}, {{3, 10, 5, 6}}, {{1, 2, 3, 4, 2, 3, 2, 2}}};
const std::vector<std::vector<size_t>> constShapes = {{1}};
const std::vector<size_t> levels = {16, 255, 256};

const std::vector<float> fqArgs = {};

const auto fqParams = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(constShapes),
        ::testing::Values(fqArgs),
        ::testing::Values(ov::op::AutoBroadcastType::NUMPY)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize, FakeQuantizeLayerTest,
                        ::testing::Combine(
                                fqParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        FakeQuantizeLayerTest::getTestCaseName);

}  // namespace
