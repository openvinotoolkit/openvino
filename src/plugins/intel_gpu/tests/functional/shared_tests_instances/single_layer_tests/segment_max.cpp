// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/segment_max.hpp"

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_SegmentMaxTest,
                         SegmentMaxLayerTest,
                         ::testing::Combine(::testing::ValuesIn(SegmentMaxLayerTest::GenerateParams()),
                                            testing::Values(ElementType::f32, ElementType::f16, ElementType::i32),
                                            testing::Values(ov::test::utils::DEVICE_GPU)),
                         SegmentMaxLayerTest::getTestCaseName);

}  // namespace test
}  // namespace ov
