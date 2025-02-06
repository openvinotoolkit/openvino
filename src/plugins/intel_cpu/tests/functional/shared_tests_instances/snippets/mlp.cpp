// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp.hpp"

#include "common_test_utils/test_constants.hpp"
#include "internal_properties.hpp"
#include "utils/cpu_test_utils.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<InputShape>> get_shapes_3D() {
    auto shapes = SNIPPETS_TESTS_STATIC_SHAPES(
            {{1, 128, 4096}},
            {{2, 43, 123}}
    );
    shapes.push_back({{PartialShape{2, -1, 300}, {{2, 128, 300},  {2, 70, 300}}}});
    shapes.push_back({{PartialShape{-1, -1, 31}, {{2, 13, 31}, {3, 49, 31}}}});
    return shapes;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_3D,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(get_shapes_3D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(1),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MLP::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_3D_BF16,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(get_shapes_3D()),
                                            ::testing::ValuesIn(precision_bf16_if_supported(1)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(3), // MLP subgraph + input and output Convert (also subgraphs)
                                            ::testing::Values(3),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MLP::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_3D_EnforceBF16_NoConverts,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(get_shapes_3D()),
                                            ::testing::ValuesIn(precision_bf16_if_supported(1)),
                                            ::testing::Values(ov::element::bf16),
                                            ::testing::Values(1),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MLP::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_3D_EnforceBF16,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(get_shapes_3D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::Values(ov::element::bf16),
                                            ::testing::Values(6), // 3 Subgraphs + 3 Reorders after Constants
                                            ::testing::Values(3), // MLP subgraph + input and output Convert (also subgraphs)
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MLP::getTestCaseName);


// todo: remove the comments or support 4D cases
//std::vector<std::vector<ov::test::InputShape>> inputShapes_4D_dynamic{
//        {
//            {PartialShape{-1, -1, -1, -1}, {{1, 128, 3, 64}, {1, 70, 3, 19}, {1, 128, 3, 64}, {1, 68, 6, 87}}},
//            {PartialShape{-1, -1, -1, -1}, {{1, 128, 1, 64}, {2, 49, 1, 19}, {1, 128, 1, 64}, {2, 13, 6, 87}}},
//            {PartialShape{-1, -1, -1, -1}, {{2, 1, 128, 128}, {1, 1, 70, 49}, {2, 1, 128, 128}, {1, 1, 68, 13}}},
//            {PartialShape{-1, -1, -1, -1}, {{1, 128, 3, 64}, {1, 49, 3, 19}, {1, 128, 3, 64}, {2, 13, 6, 87}}},
//        },
//        {
//            {PartialShape{-1, -1, 12, 64}, {{1, 70, 12, 64}, {1, 20, 12, 64}, {1, 20, 12, 64}, {1, 20, 12, 64}, {1, 70, 12, 64}}},
//            {PartialShape{-1, -1, 12, 64}, {{1, 35, 12, 64}, {2, 10, 12, 64}, {2, 1, 12, 64}, {2, 10, 12, 64}, {1, 35, 12, 64}}},
//            {PartialShape{-1, 12, -1, -1}, {{2, 12, 70, 35}, {1, 12, 20, 10}, {1, 12, 20, 10}, {1, 12, 20, 1},  {2, 12, 70, 35}}},
//            {PartialShape{-1, -1, 12, 64}, {{1, 35, 12, 64}, {1, 10, 12, 64}, {1, 10, 12, 64}, {1, 10, 12, 64}, {1, 35, 12, 64}}},
//        }
//};

//INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DynMHA_4D,
//                         MHA,
//                         ::testing::Combine(::testing::ValuesIn(inputShapes_4D_dynamic),
//                                            ::testing::ValuesIn(precision_f32(4)),
//                                            ::testing::Values(ov::element::f32),
//                                            ::testing::ValuesIn({false}),
//                                            ::testing::Values(MHA::default_thread_count),
//                                            ::testing::Values(1),
//                                            ::testing::Values(1),
//                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
//                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
//                         MHA::getTestCaseName);


//INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_3D,
//                         MHA,
//                         ::testing::Combine(::testing::ValuesIn(inputShapes_3D),
//                                            ::testing::ValuesIn(precision_f32(4)),
//                                            ::testing::Values(ov::element::f32),
//                                            ::testing::ValuesIn({false, true}),
//                                            ::testing::Values(MHA::default_thread_count),
//                                            ::testing::Values(5),  // [122706]: Subgraph + 4 Transpose
//                                            ::testing::Values(2),  // decomposed Transpose + MHA
//                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
//                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
//                         MHA::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
