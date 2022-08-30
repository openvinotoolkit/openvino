
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "snippets/codegen_gelu.hpp"
#include "common_test_utils/test_constants.hpp"
//  todo: Rewrite this test using Snippets test infrastructure. See add_convert or conv_eltwise for example

namespace ov {
namespace test {
namespace snippets {
namespace {

    INSTANTIATE_TEST_SUITE_P(NoReshape, CodegenGelu,
            ::testing::Combine(
            ::testing::Values(ov::element::f32),
            ::testing::Values(ov::Shape {1, 384, 4096}),
            ::testing::Values(true, false),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            CodegenGelu::getTestCaseName);
}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov