// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_api_conformance_helpers.hpp"


namespace {
using namespace ov::test::behavior;
using namespace ov::test::conformance;
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model_mandatory, OVClassCompiledModelGetPropertyTest,
        ::testing::Values(ov::test::utils::target_device));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model_mandatory, OVClassCompiledModelGetIncorrectPropertyTest,
        ::testing::Values(ov::test::utils::target_device));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model_mandatory, OVClassCompiledModelGetConfigTest,
        ::testing::Values(ov::test::utils::target_device));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassCompiledModelSetIncorrectConfigTest,
        ::testing::Values(ov::test::utils::target_device));


} // namespace

