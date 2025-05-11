// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/normalize_l2_decomposition.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeNormalizeL2, ExecGrapDecomposeNormalizeL2,
                        ::testing::Values(ov::test::utils::DEVICE_GPU),
                        ExecGrapDecomposeNormalizeL2::getTestCaseName);

} // namespace
