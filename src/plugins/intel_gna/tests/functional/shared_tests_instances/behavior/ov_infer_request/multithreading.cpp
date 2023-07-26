// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/multithreading.hpp"

using namespace ov::test::behavior;
namespace {

OPENVINO_SUPPRESS_DEPRECATED_START

const std::vector<ov::AnyMap> configs = {{{GNA_CONFIG_KEY(LIB_N_THREADS), "3"}}};

OPENVINO_SUPPRESS_DEPRECATED_END

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestMultithreadingTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestMultithreadingTests::getTestCaseName);
}  // namespace
