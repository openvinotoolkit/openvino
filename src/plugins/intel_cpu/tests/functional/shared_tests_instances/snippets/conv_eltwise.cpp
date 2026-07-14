// Copyright (C) 2018-2026 Intel Corporation
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

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
constexpr size_t convAddNumNodes = 5;  // num nodes = 5: Convert + Convolution + 3 Reorders on Convs in&outs
constexpr size_t convAddNumSubgraphs = 0;  // num subgraphs = 0: No subgraph since all ops eltwises fused into Convolution
#elif defined(OPENVINO_ARCH_ARM64)
// ARM64 inserts one additional reorder for the binary post-op input to match the convolution output layout.
constexpr size_t convAddNumNodes = 6;
constexpr size_t convAddNumSubgraphs = 1;
#else  // defined(OPENVINO_ARCH_RISCV64)
// RISCV64 keeps Convolution and Sinh separate and tokenizes Add, Abs, and Sqrt into one Subgraph.
constexpr size_t convAddNumNodes = 3;
constexpr size_t convAddNumSubgraphs = 1;
#endif

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvAdd, ConvEltwise,
        ::testing::Combine(
        ::testing::Values(convInputShape),
        ::testing::Values(convInputShape),
        ::testing::Values(std::shared_ptr<ov::Node> (std::make_shared<ov::op::v1::Add>())), // non-tokenizable
        ::testing::Values(convAddNumNodes),
        ::testing::Values(convAddNumSubgraphs),
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
