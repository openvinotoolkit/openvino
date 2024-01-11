// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/exec_graph_info.hpp"

#include "behavior/ov_plugin/properties_tests.hpp"

namespace {
using ov::test::behavior::OVExecGraphUniqueNodeNames;

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, OVExecGraphUniqueNodeNames,
        ::testing::Combine(
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::Shape{1, 2, 5, 5}),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
        OVExecGraphUniqueNodeNames::getTestCaseName);
}  // namespace
