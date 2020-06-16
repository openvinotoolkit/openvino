// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request_config.hpp"

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values("TEMPLATE"),
                                    ::testing::ValuesIn(configs)),
                            InferConfigTests::getTestCaseName);
}  // namespace
