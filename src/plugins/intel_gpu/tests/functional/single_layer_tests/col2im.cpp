// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"

#include "shared_test_classes/single_op/col2im.hpp"

using ov::test::InputShape;
using ov::test::Col2Im::Col2ImOpsSpecificParams;
using ov::test::Col2Im::Col2ImLayerSharedTestParams;

namespace ov {
namespace test {
namespace Col2Im {
TEST_P(Col2ImLayerSharedTest, Col2ImCompareWithRefs) {
    run();
};

INSTANTIATE_TEST_SUITE_P(smoke_Col2ImLayerGPUTestNoneBatch,
    Col2ImLayerSharedTest,
    ::testing::Combine(::testing::Combine(::testing::Values(ov::Shape{12, 9}),
                                          ::testing::Values(std::vector<int64_t>{4, 4}),
                                          ::testing::Values(std::vector<int64_t>{2, 2}),
                                          ::testing::Values(ov::Strides{1, 1}),
                                          ::testing::Values(ov::Strides{1, 1}),
                                          ::testing::Values(ov::Shape{0, 0}),
                                          ::testing::Values(ov::Shape{0, 0})),
                       ::testing::Values(ov::element::f16),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                       Col2ImLayerSharedTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Col2ImLayerGPUTestNoneBatch3D,
Col2ImLayerSharedTest,
::testing::Combine(::testing::Combine(::testing::Values(ov::Shape{1, 12, 9}),
                                        ::testing::Values(std::vector<int64_t>{4, 4}),
                                        ::testing::Values(std::vector<int64_t>{2, 2}),
                                        ::testing::Values(ov::Strides{1, 1}),
                                        ::testing::Values(ov::Strides{1, 1}),
                                        ::testing::Values(ov::Shape{0, 0}),
                                        ::testing::Values(ov::Shape{0, 0})),
                    ::testing::Values(ov::element::f16),
                    ::testing::Values(ov::element::i32),
                    ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                    Col2ImLayerSharedTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Col2ImLayerGPUTestPadded,
Col2ImLayerSharedTest,
::testing::Combine(::testing::Combine(::testing::Values(ov::Shape{12, 81}),
                                        ::testing::Values(std::vector<int64_t>{16, 16}),
                                        ::testing::Values(std::vector<int64_t>{2, 2}),
                                        ::testing::Values(ov::Strides{2, 2}),
                                        ::testing::Values(ov::Strides{2, 2}),
                                        ::testing::Values(ov::Shape{2, 2}),
                                        ::testing::Values(ov::Shape{2, 2})),
                    ::testing::Values(ov::element::f16),
                    ::testing::Values(ov::element::i32),
                    ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                    Col2ImLayerSharedTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Col2ImLayerGPUTestFp16,
Col2ImLayerSharedTest,
::testing::Combine(::testing::Combine(::testing::Values(ov::Shape{1, 12, 9}),
                                        ::testing::Values(std::vector<int64_t>{4, 4}),
                                        ::testing::Values(std::vector<int64_t>{2, 2}),
                                        ::testing::Values(ov::Strides{1, 1}),
                                        ::testing::Values(ov::Strides{1, 1}),
                                        ::testing::Values(ov::Shape{0, 0}),
                                        ::testing::Values(ov::Shape{0, 0})),
                    ::testing::Values(ov::element::f16),
                    ::testing::Values(ov::element::i32),
                    ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                    Col2ImLayerSharedTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Col2ImLayerGPUTest_Batch,
    Col2ImLayerSharedTest,
    ::testing::Combine(::testing::Combine(::testing::Values(ov::Shape{12, 12, 324}),
                                            ::testing::Values(std::vector<int64_t>{32, 32}),
                                            ::testing::Values(std::vector<int64_t>{2, 2}),
                                            ::testing::Values(ov::Strides{2, 2}),
                                            ::testing::Values(ov::Strides{2, 2}),
                                            ::testing::Values(ov::Shape{3, 3}),
                                            ::testing::Values(ov::Shape{3, 3})),
                        ::testing::Values(ov::element::f16),
                        ::testing::Values(ov::element::i32),
                        ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        Col2ImLayerSharedTest::getTestCaseName);

}  // namespace Col2Im
}  // namespace test
}  // namespace ov
