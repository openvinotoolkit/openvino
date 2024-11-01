// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include "behavior/ov_plugin/core_threading.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"

namespace {

const Params params[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_NPU, {{ov::enable_profiling(true)}}},
    std::tuple<Device, Config>{ov::test::utils::DEVICE_HETERO,
                               {{ov::device::priorities(ov::test::utils::DEVICE_NPU, ov::test::utils::DEVICE_CPU)}}},
};

const Params paramsStreams[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_NPU, {{ov::num_streams(ov::streams::AUTO)}}},
};

const Params paramsStreamsDRIVER[] = {
    std::tuple<Device, Config>{
        ov::test::utils::DEVICE_NPU,
        {{ov::num_streams(ov::streams::AUTO), ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}}},
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_CoreThreadingTest_NPU,
                         CoreThreadingTest,
                         testing::ValuesIn(params),
                         (ov::test::utils::appendPlatformTypeTestName<CoreThreadingTest>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CoreThreadingTest_NPU,
                         CoreThreadingTestsWithIter,
                         testing::Combine(testing::ValuesIn(params), testing::Values(4), testing::Values(50)),
                         (ov::test::utils::appendPlatformTypeTestName<CoreThreadingTestsWithIter>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CoreThreadingTest_Streams_NPU,
                         CoreThreadingTestsWithCacheEnabled,
                         testing::Combine(testing::ValuesIn(paramsStreamsDRIVER),
                                          testing::Values(20),
                                          testing::Values(10)),
                         (ov::test::utils::appendPlatformTypeTestName<CoreThreadingTestsWithCacheEnabled>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CoreThreadingTest_Streams_NPU,
                         CoreThreadingTestsWithIter,
                         testing::Combine(testing::ValuesIn(paramsStreams), testing::Values(4), testing::Values(50)),
                         (ov::test::utils::appendPlatformTypeTestName<CoreThreadingTestsWithIter>));
