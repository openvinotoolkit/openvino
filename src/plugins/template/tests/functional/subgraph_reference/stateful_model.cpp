// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/stateful_model.hpp"

using namespace ov::test;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke, StatefulModelStateInLoopBody, ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

}  // namespace

// Other stateful model suites are defined via TEST_P in the shared header but
// not instantiated by the template plugin.
namespace ov::test {
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(StaticShapeStatefulModel);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(StaticShapeTwoStatesModel);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DynamicShapeStatefulModelDefault);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DynamicShapeStatefulModelParam);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DynamicShapeStatefulModelStateAsInp);
}  // namespace ov::test
