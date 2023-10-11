// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "behavior/executable_network/exec_graph_info.hpp"

namespace {

using namespace ExecutionGraphTests;

INSTANTIATE_TEST_SUITE_P(smoke_serialization, ExecGraphSerializationTest,
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                        ExecGraphSerializationTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ExecGraphUniqueNodeNames,
        ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({1, 2, 5, 5})),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
        ExecGraphUniqueNodeNames::getTestCaseName);

}  // namespace

