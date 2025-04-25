// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/stft.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_STFT_static,
                         STFTLayerTest,
                         STFTLayerTest::GetTestDataForDevice(ov::test::utils::DEVICE_CPU),
                         STFTLayerTest::getTestCaseName);
}  // namespace test
}  // namespace ov
