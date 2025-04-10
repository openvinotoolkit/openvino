// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
// Subgraph:
/*
 *          paramter1         parameter2
 *                  \             /
 *                   \           /
 *                   ReLu1    ReLu2
 *                   / \       /
 *                  /   \     /
 *                 /     \   /
 *                 |     Concat (inPlace)
 *                 |       |
 *                 |      Conv
 *                 |       |
 *                 |      Add_Const (Bias)
 *                  \      |
 *                   \     |
 *                    \    |
 *                     \   |
 *                       Sum
 *                        |
 *                       ReLu3
 *                        |
 *                      Result
 */

class ReLuConcatConvSumInPlaceTest : virtual public SubgraphBaseStaticTest {
public:
    void SetUp() override {
        const ov::Shape inputShape = {1, 64, 12, 12};
        const std::vector<size_t> kernel = {1, 1};
        const std::vector<size_t> stride = {1, 1};
        const std::vector<size_t> dilation = {1, 1};
        const std::vector<ptrdiff_t> padBegin = {0, 0};
        const std::vector<ptrdiff_t> padEnd = {0, 0};
        const size_t convOutChannels = 64;
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        const auto targetFormat = with_cpu_x86_avx512_core() ? nChw16c : nChw8c;
#else
        const auto targetFormat = nhwc;
#endif

        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape),
                                        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape)};
        auto Relu1 = std::make_shared<ov::op::v0::Relu>(inputParams[0]);
        Relu1->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto Relu2 = std::make_shared<ov::op::v0::Relu>(inputParams[1]);
        Relu2->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{Relu1, Relu2}, 1);

        auto conv = ov::test::utils::make_convolution(concat,
                                                      ov::element::f32,
                                                      kernel,
                                                      stride,
                                                      padBegin,
                                                      padEnd,
                                                      dilation,
                                                      ov::op::PadType::AUTO,
                                                      convOutChannels);
        auto bias = ov::test::utils::make_constant(ov::element::Type_t::f32, ov::Shape({1, convOutChannels, 1, 1}));
        auto convBiasAdd = std::make_shared<ov::op::v1::Add>(conv, bias);

        auto sum = std::make_shared<ov::op::v1::Add>(convBiasAdd, Relu1);

        auto Relu3 = std::make_shared<ov::op::v0::Relu>(sum);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Relu3)};
        function = std::make_shared<ov::Model>(results, inputParams, "ConcatConvSumInPlace");
        targetDevice = ov::test::utils::DEVICE_CPU;
    }
};

namespace {
TEST_F(ReLuConcatConvSumInPlaceTest, smoke_ReLuConcatConvSumInPlace_CPU) {
    run();
}
}  // namespace
}  // namespace test
}  // namespace ov
