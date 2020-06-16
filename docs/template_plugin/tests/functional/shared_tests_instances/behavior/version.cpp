// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/version.hpp"
using namespace BehaviorTestsUtils;
namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, VersionTest,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values("TEMPLATE"),
                                    ::testing::ValuesIn(configs)),
                            VersionTest::getTestCaseName);

}  // namespace
