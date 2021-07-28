// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/detect_network_batch_test.hpp"

using namespace LayerTestsDefinitions;

const std::vector<unsigned int> batchSizes = {
    2,
    4,
    8,
};

namespace {
    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, DetectNetworkBatch,
                            ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::ValuesIn(batchSizes)),
                            DetectNetworkBatch::getTestCaseName);
}  // namespace
