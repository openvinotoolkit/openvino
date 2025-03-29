// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

namespace {
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values("MULTI:TEMPLATE", "AUTO:TEMPLATE"));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetIncorrectPropertyTest,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values("MULTI:TEMPLATE", "AUTO:TEMPLATE"));
//////////////////////////////////////////////////////////////////////////////////////////

}  // namespace
