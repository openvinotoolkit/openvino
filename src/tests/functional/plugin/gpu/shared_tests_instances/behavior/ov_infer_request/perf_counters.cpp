// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/perf_counters.hpp"

using namespace ov::test::behavior;

namespace {
TEST_P(OVInferRequestPerfCountersTest, CheckOperationInProfilingInfo) {
    req = execNet.create_infer_request();
    ASSERT_NO_THROW(req.infer());

    std::vector<ov::ProfilingInfo> profiling_info;
    ASSERT_NO_THROW(profiling_info = req.get_profiling_info());

    for (const auto& op : function->get_ops()) {
        auto op_is_in_profiling_info = std::any_of(std::begin(profiling_info), std::end(profiling_info),
            [&] (const ov::ProfilingInfo& info) {
            if (info.node_name.find(op->get_friendly_name() + "_") != std::string::npos || info.node_name == op->get_friendly_name()) {
                return true;
            } else {
                return false;
            }
        });
        ASSERT_TRUE(op_is_in_profiling_info) << "For op: " << op;
    }
}

const std::vector<ov::AnyMap> configs = {
        {}
};

const std::vector<ov::AnyMap> Multiconfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU)}
};

const std::vector<ov::AnyMap> Autoconfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU)}
};

const std::vector<ov::AnyMap> AutoBatchConfigs = {
        // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , std::string(CommonTestUtils::DEVICE_GPU) + "(4)"},
                // no timeout to avoid increasing the test time
                {CONFIG_KEY(AUTO_BATCH_TIMEOUT) , "0 "}}
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
