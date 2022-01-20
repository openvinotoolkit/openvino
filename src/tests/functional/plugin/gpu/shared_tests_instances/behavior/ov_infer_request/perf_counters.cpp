// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/perf_counters.hpp"

using namespace ov::test::behavior;

namespace {
TEST_P(OVInferRequestPerfCountersTest, CheckOperationInProfilingInfo) {
    req = execNet.create_infer_request();
    ASSERT_NO_THROW(req.infer());

    std::vector<ov::runtime::ProfilingInfo> profiling_info;
    ASSERT_NO_THROW(profiling_info = req.get_profiling_info());

    for (const auto& op : function->get_ops()) {
        auto op_is_in_profiling_info = std::any_of(std::begin(profiling_info), std::end(profiling_info),
            [&] (const ov::runtime::ProfilingInfo& info) {
            if (info.node_name.find(op->get_friendly_name() + "_") != std::string::npos || info.node_name == op->get_friendly_name()) {
                return true;
            } else {
                return false;
            }
        });
        ASSERT_TRUE(op_is_in_profiling_info) << "For op: " << op;
    }
}

const std::vector<std::map<std::string, std::string>> configs = {
        {}
};

const std::vector<std::map<std::string, std::string>> Multiconfigs = {
        {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_GPU}}
};

const std::vector<std::map<std::string, std::string>> Autoconfigs = {
        {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_GPU}}
};

const std::vector<std::map<std::string, std::string>> AutoBatchConfigs = {
        {{ CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                ::testing::ValuesIn(configs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(Multiconfigs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(Autoconfigs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVInferRequestPerfCountersTest,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                 ::testing::ValuesIn(AutoBatchConfigs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);
}  // namespace
