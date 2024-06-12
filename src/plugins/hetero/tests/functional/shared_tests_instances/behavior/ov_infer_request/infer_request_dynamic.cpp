// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"

#include <vector>

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> HeteroConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Hetero_BehaviorTests,
    OVInferRequestDynamicTests,
    ::testing::Combine(::testing::Values(ov::test::utils::make_split_conv_concat()),
                       ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                           {{1, 4, 20, 20}, {1, 10, 18, 18}},
                           {{2, 4, 20, 20}, {2, 10, 18, 18}}}),
                       ::testing::Values(ov::test::utils::DEVICE_HETERO),
                       ::testing::ValuesIn(HeteroConfigs)),
    OVInferRequestDynamicTests::getTestCaseName);

}  // namespace
