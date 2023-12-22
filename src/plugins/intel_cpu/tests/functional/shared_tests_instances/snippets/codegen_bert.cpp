
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "snippets/codegen_bert.hpp"
#include "common_test_utils/test_constants.hpp"
//  todo: Rewrite this test using Snippets test infrastructure. See add_convert or conv_eltwise for example

namespace ov {
namespace test {
namespace snippets {
namespace {

    INSTANTIATE_TEST_SUITE_P(NoReshape, CodegenBert,
            ::testing::Combine(
            ::testing::Values(ov::element::f32),
            ::testing::Values(ov::Shape {1, 42, 16, 64}),
            ::testing::Values(ov::Shape {1, 42, 64, 64}),
            ::testing::Values(ov::test::utils::DEVICE_CPU)),
            CodegenBert::getTestCaseName);
}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov
