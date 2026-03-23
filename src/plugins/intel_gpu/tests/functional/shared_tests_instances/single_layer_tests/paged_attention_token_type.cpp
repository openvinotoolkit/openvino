// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_attention_token_type.hpp"

// #include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
// #include "common_test_utils/node_builders/constant.hpp"
// #include "openvino/core/type/float16.hpp"
// #include "openvino/op/paged_attention.hpp"
// #include "openvino/op/parameter.hpp"
// #include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov::test;
using namespace ov::op;

namespace ov {
namespace test {

TEST_P(PagedAttentionTokenTypeTest, ImageTokensDifferFromCausal) {
    RunAndValidate();
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttentionTokenType,
                         PagedAttentionTokenTypeTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::Values(32),  // head_size
                                            ::testing::Values(1),   // head_num
                                            ::testing::ValuesIn(PagedAttentionTokenTypeTest::GetTestData()),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         PagedAttentionTokenTypeTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
