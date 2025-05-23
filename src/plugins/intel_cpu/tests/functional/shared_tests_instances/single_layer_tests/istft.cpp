// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/istft.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_ISTFT,
                         ISTFTLayerTest,
                         ISTFTLayerTest::GetTestDataForDevice(ov::test::utils::DEVICE_CPU),
                         ISTFTLayerTest::getTestCaseName);
}  // namespace test
}  // namespace ov
