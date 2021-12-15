// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ie_precision.hpp>
#include <subgraph_tests/cascade_concat.hpp>
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    std::vector<std::vector<std::vector<size_t>>> shape1{
            {{1, 64}},
            {{1, 128}},
            {{1, 32}},
            {{1, 16}},
            {{1, 8}}
    };

    std::vector<std::vector<std::vector<size_t>>> shape2{
            {{1, 72}},
            {{1, 128}},
            {{1, 32}},
            {{1, 16}},
            {{1, 8}}
    };

    std::vector<std::vector<std::vector<size_t>>> shape3{
            {{1, 80}},
            {{1, 128}},
            {{1, 32}},
            {{1, 16}},
            {{1, 8}}
    };

    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
    };

    std::map<std::string, std::string> additional_config = {
            {"GNA_SCALE_FACTOR_0", "1"},
            {"GNA_SCALE_FACTOR_1", "1"},
            {"GNA_SCALE_FACTOR_2", "1"}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_cascade_concat, CascadeConcat,
                            ::testing::Combine(
                                    ::testing::ValuesIn(shape1),
                                    ::testing::ValuesIn(shape2),
                                    ::testing::ValuesIn(shape3),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(false),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::Values(additional_config)),
                            CascadeConcat::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_cascade_concat_multioutput, CascadeConcat,
                            ::testing::Combine(
                                    ::testing::ValuesIn(shape1),
                                    ::testing::ValuesIn(shape2),
                                    ::testing::ValuesIn(shape3),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(true),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::Values(additional_config)),
                            CascadeConcat::getTestCaseName);
}  // namespace
