// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
typedef std::tuple<
        InputShape, //convShape
        InputShape,  //second term shape
        bool,       // bias flag
        fusingSpecificParams
> convSumBroadcastParamSet;


class ConcatConvSumInPlaceTest : public testing::WithParamInterface<convSumBroadcastParamSet>,
                                 virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convSumBroadcastParamSet>& obj) {
        InputShape convShape;
        InputShape secondShape;
        bool bias;
        fusingSpecificParams fusingParams;
        std::tie(convShape, secondShape, bias, fusingParams) = obj.param;

        std::ostringstream result;
        result << "IS=";
        result  << CommonTestUtils::partialShape2str({convShape.first, secondShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : {convShape, secondShape}) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << CommonTestUtils::vec2str(itr);
                }
            }
            result << ")_";
        }
        result << "bias=" << (bias ? "True" : "False");
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

    void SetUp() override {
        InputShape convShape;
        InputShape secondShape;
        bool bias;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(convShape, secondShape, bias, fusingParams) = this->GetParam();

        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        init_input_shapes({convShape, secondShape});

        const InferenceEngine::SizeVector kernel = {3, 3};
        const InferenceEngine::SizeVector stride = {1, 1};
        const InferenceEngine::SizeVector dilation = {1, 1};
        const std::vector<ptrdiff_t> padBegin = {0, 0};
        const std::vector<ptrdiff_t> padEnd = {0, 0};
        const size_t convOutChannels = 64;

        auto netType = ngraph::element::f32;
        auto inputParams = ngraph::builder::makeDynamicParams(netType, inputDynamicShapes);

        auto conv = ngraph::builder::makeConvolution(inputParams[0], ngraph::element::f32, kernel, stride, padBegin,
                                                     padEnd, dilation, ngraph::op::PadType::EXPLICIT, convOutChannels);
        if (bias) {
            auto biasNode = ngraph::builder::makeConstant<float>(ngraph::element::Type_t::f32, ngraph::Shape({1, convOutChannels, 1, 1}), {}, true);
            conv = std::make_shared<ngraph::opset3::Add>(conv, biasNode);
        }

        auto sum = std::make_shared<ngraph::opset3::Add>(conv, inputParams[1]);

        selectedType = makeSelectedTypeStr(getPrimitiveType(), netType);

        function = makeNgraphFunction(netType, inputParams, sum, "ConvolutionSumBroadcast");
        targetDevice = CommonTestUtils::DEVICE_CPU;
    }
};

TEST_P(ConcatConvSumInPlaceTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();

    CheckPluginRelatedResults(executableNetwork, "Convolution");
}

namespace {
const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
        fusingSigmoid,
        fusingFakeQuantizePerTensorRelu,
        fusingFakeQuantizePerChannelRelu,
        fusingReluScaleShift
};

InputShape convInpShape = {
        //dynamic shapes
        {-1, 32, -1, -1},
        { //target static shapes
            {1, 32, 10, 10},
            {1, 32, 10, 10},
            {1, 32, 10, 10},
            {1, 32, 3, 3}
        }
};

InputShape secondInp = {
        //dynamic shapes
        {-1, -1, -1, -1},
        { //target static shapes
            {1, 64, 1, 8},
            {1, 64, 1, 8},
            {1, 64, 8, 8},
            {1, 64, 8, 8}
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast, ConcatConvSumInPlaceTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape),
                                 ::testing::Values(secondInp),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSet)),
                         ConcatConvSumInPlaceTest::getTestCaseName);

} // namespace
} // namespace SubgraphTestsDefinitions
