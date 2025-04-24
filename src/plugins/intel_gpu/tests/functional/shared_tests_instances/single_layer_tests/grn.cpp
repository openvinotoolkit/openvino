// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_op_tests/grn.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::GrnLayerTest;
// Common params
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{1, 3, 30, 30}},
    {{2, 16, 15, 20}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Grn_Basic,
                        GrnLayerTest,
                        ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                ::testing::ValuesIn({0.33f, 1.1f}),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GrnLayerTest::getTestCaseName);

}  // namespace
