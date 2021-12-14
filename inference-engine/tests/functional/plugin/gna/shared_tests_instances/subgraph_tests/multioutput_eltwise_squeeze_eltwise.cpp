// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "subgraph_tests/multioutput_eltwise_squeeze_eltwise.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    std::vector<std::vector<std::vector<size_t>>> inputs{
            {{1, 16}},
            {{2, 16}},
            {{1, 160}},
            {{8, 40}},
            {{3, 8}},
            {{4, 32}},
            {{5, 64}},
            {{6, 128}},
            {{7, 256}},
            {{8, 512}},
            {{8, 1024}}
    };

    std::map<std::string, std::string> additional_config = {
            {"GNA_COMPACT_MODE", "NO"},
    };

    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16,
    };

    INSTANTIATE_TEST_SUITE_P(smoke_multioutput_eltwise_identity, MultioutputEltwiseReshapeEltwise,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inputs),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::Values(additional_config)),
                            MultioutputEltwiseReshapeEltwise::getTestCaseName);
}  // namespace
