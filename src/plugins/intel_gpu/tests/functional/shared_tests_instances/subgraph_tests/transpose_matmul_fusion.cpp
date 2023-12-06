// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/transpose_matmul_fusion.hpp"

using namespace ov::test;

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_TransposeMatMulFusion, TransposeMatMulFusion,
                         ::testing::Values(ov::test::utils::DEVICE_GPU),
                         TransposeMatMulFusion::getTestCaseName);

}  // namespace
