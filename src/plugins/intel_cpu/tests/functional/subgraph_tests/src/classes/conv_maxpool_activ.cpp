// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_maxpool_activ.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

std::string ConvPoolActivTest::getTestCaseName(testing::TestParamInfo<ConvPoolActivTestParams> obj) {
    fusingSpecificParams fusingParams = obj.param;

    std::ostringstream result;
    result << "ConvPoolActivTest";
    result << CpuTestWithFusing::getTestCaseName(fusingParams);

    return result.str();
}

void ConvPoolActivTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    fusingSpecificParams fusingParams = this->GetParam();
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 40, 40})};

    std::shared_ptr<Node> conv;
    {
        const std::vector<size_t> kernelSize = {3, 3};
        const std::vector<size_t> strides = {2, 1};
        const std::vector<ptrdiff_t> padBegin = {0, 0};
        const std::vector<ptrdiff_t> padEnd = {0, 0};
        const std::vector<size_t> dilation = {1, 1};
        const size_t numOutChannels = 16;
        const op::PadType paddingType = op::PadType::EXPLICIT;
        conv = builder::makeConvolution(inputParams[0], element::f32, kernelSize, strides, padBegin, padEnd, dilation, paddingType, numOutChannels);
    }
    std::shared_ptr<Node> pooling;
    {
        const std::vector<size_t> kernelSize = {3, 3};
        const std::vector<size_t> strides = {1, 1};
        const std::vector<size_t> padBegin = {0, 0};
        const std::vector<size_t> padEnd = {0, 0};
        const op::PadType paddingType = op::PadType::EXPLICIT;
        ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::CEIL;
        pooling = std::make_shared<ov::op::v1::MaxPool>(conv, strides, padBegin, padEnd, kernelSize, roundingType, paddingType);
    }

    function = makeNgraphFunction(element::f32, inputParams, pooling, "ConvPoolActiv");
}

TEST_P(ConvPoolActivTest, CompareWithRefs) {
    run();
}

} // namespace SubgraphTestsDefinitions
