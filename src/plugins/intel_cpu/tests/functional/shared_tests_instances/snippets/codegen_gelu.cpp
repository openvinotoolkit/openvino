
// Copyright (C) 2018-2025 Intel Corporation
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
std::vector<InputShape> inShapes0{
    {{}, {{1, 38, 130}}},
    {{}, {{1, 1, 130}}},
};
std::vector<InputShape> inShapes1{
    {{}, {{1, 38, 130}}},
    {{}, {{1, 38, 1}}},
};

INSTANTIATE_TEST_SUITE_P(NoReshapeAndReshape, CodegenGelu,
        ::testing::Combine(
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(inShapes0),
        ::testing::ValuesIn(inShapes1),
        ::testing::Values(true, false),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
        CodegenGelu::getTestCaseName);

// DS
std::vector<InputShape> inShapesDynamic0{
    {{-1, -1, -1}, {{1, 12, 128}, {1, 12, 1}, {1, 12, 128}}},
};
std::vector<InputShape> inShapesDynamic1{
    {{-1, -1, -1}, {{1, 12, 128}, {1, 1, 128}, {1, 12, 128}}},
};

INSTANTIATE_TEST_SUITE_P(NoReshapeAndReshapeDynamic, CodegenGelu,
        ::testing::Combine(
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(inShapesDynamic0),
        ::testing::ValuesIn(inShapesDynamic1),
        ::testing::Values(true, false),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
        CodegenGelu::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov
