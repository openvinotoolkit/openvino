// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/transpose_matmul_fusion.hpp"

using namespace ov::test;

namespace {
// This test is temporarily disabled due to different compiled model between CPU and GPU.
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_TransposeMatMulFusion, TransposeMatMulFusion,
                         ::testing::Values(ov::test::utils::DEVICE_GPU),
                         TransposeMatMulFusion::getTestCaseName);

}  // namespace
