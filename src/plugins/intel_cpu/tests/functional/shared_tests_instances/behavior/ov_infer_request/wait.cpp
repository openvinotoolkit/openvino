// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/wait.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configs = {{},
                                         {{ov::num_streams(ov::streams::AUTO)}},
                                         {{ov::num_streams(ov::streams::Num(0))}, {ov::inference_num_threads(1)}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestWaitTests::getTestCaseName);
}  // namespace
