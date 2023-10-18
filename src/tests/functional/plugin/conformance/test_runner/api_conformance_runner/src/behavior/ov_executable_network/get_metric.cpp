// Copyright (C) 2018-2023 Intel Corporation
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
        ::testing::ValuesIn(return_batch_combination()));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model_mandatory, OVClassCompiledModelGetIncorrectPropertyTest,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model_mandatory, OVClassCompiledModelGetConfigTest,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassCompiledModelSetIncorrectConfigTest,
        ::testing::ValuesIn(return_all_possible_device_combination()));

} // namespace

