// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/import_export.hpp"

#include "behavior/ov_plugin/properties_tests.hpp"

namespace {
using ov::test::behavior::OVCompiledModelGraphUniqueNodeNamesTest;

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, OVCompiledModelGraphUniqueNodeNamesTest,
        ::testing::Combine(
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::Shape{1, 2, 5, 5}),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
        OVCompiledModelGraphUniqueNodeNamesTest::getTestCaseName);
}  // namespace
