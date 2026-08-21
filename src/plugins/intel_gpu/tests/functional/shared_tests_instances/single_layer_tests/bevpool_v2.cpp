// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/bevpool_v2.hpp"

#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_BevPoolV2,
                         BevPoolV2LayerTest,
                         BevPoolV2LayerTest::GetTestDataForDevice(ov::test::utils::DEVICE_GPU),
                         BevPoolV2LayerTest::getTestCaseName);

}  // namespace test
}  // namespace ov
