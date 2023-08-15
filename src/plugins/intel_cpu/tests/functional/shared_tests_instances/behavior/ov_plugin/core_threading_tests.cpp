// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/ov_plugin/core_threading.hpp>

namespace {
const Params paramsStreams[] = {
    std::tuple<Device, Config>{ ov::test::utils::DEVICE_CPU, {{ov::num_streams(ov::streams::AUTO)}}},
};
}  // namespace

INSTANTIATE_TEST_SUITE_P(CPU_Streams, CoreThreadingTestsWithCacheEnabled,
    testing::Combine(testing::ValuesIn(paramsStreams),
                     testing::Values(20),
                     testing::Values(10)),
    CoreThreadingTestsWithCacheEnabled::getTestCaseName);
