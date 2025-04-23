// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/conv_eltwise.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

ov::Shape convInputShape {1, 10, 16, 16};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvAdd, ConvEltwise,
        ::testing::Combine(
        ::testing::Values(convInputShape),
        ::testing::Values(convInputShape),
        ::testing::Values(std::shared_ptr<ov::Node> (std::make_shared<ov::op::v1::Add>())), // non-tokenizable
        ::testing::Values(5), // num nodes = 5: Convert + Convolution + 3 Reorders on Convs in&outs
        ::testing::Values(0), // num subgraphs = 0: No subgraph since all ops eltwises fused into Convolution
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ConvEltwise::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvMul, ConvEltwise,
        ::testing::Combine(
        ::testing::Values(convInputShape),
        ::testing::Values(convInputShape),
        ::testing::Values(std::shared_ptr<ov::Node> (std::make_shared<ov::op::v1::Multiply>())), // fully-tokenizable
        ::testing::Values(6), //num nodes = 6: Convert + Convolution + Subgraph + Reorders
        ::testing::Values(1), // num subgraphs = 1: Mul (2 inputs) can't be fused into Conv => Subgraph is created
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ConvEltwise::getTestCaseName);
}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov
