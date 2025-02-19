// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/stridedslice_eltwise.hpp"

namespace {
using ov::test::StridedSliceEltwiseTest;

std::vector<ov::test::StridedSliceEltwiseSpecificParams> ss_spec = {
        ov::test::StridedSliceEltwiseSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 1, 2, 2, 3}, { 1, 1, 3}})),
                                { 0 }, { 2 }, { 1 },
                                { 0 }, { 0 }, { 0 }, { 1 }, { 0 } },
        ov::test::StridedSliceEltwiseSpecificParams{ ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({
                                { 2, 3, 4, 5}, { 3, 1, 1}})),
                                { 0 }, { 2 }, { 1 },
                                { 0 }, { 0 }, { 0 }, { 1 }, { 0 } },
};

INSTANTIATE_TEST_SUITE_P(smoke_StridedSliceEltwise, StridedSliceEltwiseTest,
                        testing::Combine(
                                testing::ValuesIn(ss_spec),
                                testing::Values(ov::element::f32),
                                testing::Values(ov::test::utils::DEVICE_GPU)),
                        StridedSliceEltwiseTest::getTestCaseName);

}  // namespace
