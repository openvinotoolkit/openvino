// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/matmul_transpose_to_reshape.hpp"

#include <vector>

namespace {

std::vector<ov::element::Type> netPrecisions = {ov::element::f32};

}  // namespace

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_basic,
                         MatMulTransposeToReshape,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         MatMulTransposeToReshape::getTestCaseName);

}  // namespace test
}  // namespace ov
