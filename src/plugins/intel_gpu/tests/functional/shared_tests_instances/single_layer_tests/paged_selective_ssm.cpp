// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/paged_selective_ssm.hpp"

namespace ov::test {

std::vector<PagedSelectiveSSMLayerParams> paged_selective_ssm_test_cases = {
    {4, 2, 8, 8, {3, 2}, {2, 0}, ov::element::f32, "GPU"},
    {4, 1, 8, 16, {2, 4, 1}, {1, 3, 2}, ov::element::f32, "GPU"},
    {4, 2, 8, 8, {3, 2}, {2, 0}, ov::element::f16, "GPU"},
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedSelectiveSSM,
                         PagedSelectiveSSMLayerTest,
                         ::testing::ValuesIn(paged_selective_ssm_test_cases),
                         PagedSelectiveSSMLayerTest::getTestCaseName);

}  // namespace ov::test
