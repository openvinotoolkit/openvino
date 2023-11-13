// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;

namespace SubgraphTestsDefinitions {
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

class ReLuConcatConvSumInPlaceTest : virtual public LayerTestsUtils::LayerTestsCommon {
public:
    void SetUp() override {
        const std::vector<size_t> inputShape = {1, 64, 12, 12};
        const InferenceEngine::SizeVector kernel = {1, 1};
        const InferenceEngine::SizeVector stride = {1, 1};
        const InferenceEngine::SizeVector dilation = {1, 1};
        const std::vector<ptrdiff_t> padBegin = {0, 0};
        const std::vector<ptrdiff_t> padEnd = {0, 0};
        const size_t convOutChannels = 64;
        const auto targetFormat = with_cpu_x86_avx512_core() ? nChw16c : nChw8c;


        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShape)),
                                        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShape))};
        auto Relu1 = std::make_shared<ngraph::opset3::Relu>(inputParams[0]);
        Relu1->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto Relu2 = std::make_shared<ngraph::opset3::Relu>(inputParams[1]);
        Relu2->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});

        auto concat = ngraph::builder::makeConcat(ngraph::OutputVector{Relu1, Relu2}, 1);

        auto conv = ngraph::builder::makeConvolution(concat, ngraph::element::f32, kernel, stride, padBegin,
                                                     padEnd, dilation, ngraph::op::PadType::AUTO, convOutChannels);
        auto bias = ngraph::builder::makeConstant<float>(ngraph::element::Type_t::f32, ngraph::Shape({1, convOutChannels, 1, 1}), {}, true);
        auto convBiasAdd = std::make_shared<ngraph::opset3::Add>(conv, bias);

        auto sum = std::make_shared<ngraph::opset3::Add>(convBiasAdd, Relu1);

        auto Relu3 = std::make_shared<ngraph::opset3::Relu>(sum);

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(Relu3)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "ConcatConvSumInPlace");
        targetDevice = ov::test::utils::DEVICE_CPU;
    }
};

namespace {
    TEST_F(ReLuConcatConvSumInPlaceTest, smoke_ReLuConcatConvSumInPlace_CPU) {
        Run();
    }
} // namespace
} // namespace SubgraphTestsDefinitions
