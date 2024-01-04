// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/compiled_model_base.hpp"

using namespace ov::test::behavior;
namespace {

    const std::vector<ov::AnyMap> configs = {
            {},
    };

    const std::vector<ov::AnyMap> heteroConfigs = {
            {ov::device::priorities(ov::test::utils::DEVICE_CPU)}};

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompiledModelBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                            OVCompiledModelBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVCompiledModelBaseTest,
                             ::testing::Combine(
                                     ::testing::Values(ov::test::utils::DEVICE_HETERO),
                                     ::testing::ValuesIn(heteroConfigs)),
                             OVCompiledModelBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompiledModelBaseTestOptional,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                            OVCompiledModelBaseTestOptional::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVCompiledModelBaseTestOptional,
                             ::testing::Combine(
                                     ::testing::Values(ov::test::utils::DEVICE_HETERO),
                                     ::testing::ValuesIn(heteroConfigs)),
                             OVCompiledModelBaseTestOptional::getTestCaseName);

}  // namespace
