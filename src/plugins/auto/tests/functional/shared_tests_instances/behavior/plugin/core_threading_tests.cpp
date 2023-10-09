// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/plugin/core_threading.hpp>
#ifdef __GLIBC__
#    include <gnu/libc-version.h>
#    if __GLIBC_MINOR__ >= 34
#        define ENABLETESTMULTI
#    endif
#else
#    define ENABLETESTMULTI
#endif

namespace {

const Params params[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_TEMPLATE, {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}}},
#ifdef ENABLETESTMULTI
    std::tuple<Device, Config>{ov::test::utils::DEVICE_MULTI,
                               {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::utils::DEVICE_TEMPLATE}}},
    std::tuple<Device, Config>{ov::test::utils::DEVICE_AUTO,
                               {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::utils::DEVICE_TEMPLATE}}},
#endif
};
}  // namespace
/*
INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, CoreThreadingTests, testing::ValuesIn(params),
CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(params),
                     testing::Values(4),
                     testing::Values(50),
                     testing::Values(ModelClass::Default)),
    CoreThreadingTestsWithIterations::getTestCaseName);
*/
