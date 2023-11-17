// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/stridedslice_concat.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string SliceConcatTest::getTestCaseName(const testing::TestParamInfo<SliceConcatParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    StridedSliceParams sliceParams;
    std::tie(netPrecision, targetDevice, configuration, sliceParams) = obj.param;
    std::vector<int64_t> inputShape, begin, end, strides, beginMask, endMask;
    std::tie(inputShape, begin, end, strides, beginMask, endMask) = sliceParams;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "B=" << ov::test::utils::vec2str(begin) << "_";
    result << "E=" << ov::test::utils::vec2str(end) << "_";
    result << "S=" << ov::test::utils::vec2str(strides) << "_";
    result << "BM=" << ov::test::utils::vec2str(beginMask) << "_";
    result << "EM=" << ov::test::utils::vec2str(endMask) << "_";
    return result.str();
}

void SliceConcatTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    StridedSliceParams sliceParams;
    std::tie(netPrecision, targetDevice, tempConfig, sliceParams) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());
    std::vector<int64_t> inputShape, begin, end, strides, beginMask, endMask;
    std::tie(inputShape, begin, end, strides, beginMask, endMask) = sliceParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    size_t input_size = std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<size_t>());
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, input_size})};

    ngraph::Output<ngraph::Node> input = params[0];
    if (inputShape[0] != 1 || inputShape.size() != 2) {
        input = std::make_shared<ngraph::opset8::Reshape>(params[0],
            ngraph::builder::makeConstant(ngraph::element::i64, ngraph::Shape{inputShape.size()}, inputShape), false);
    }

    auto ss = ngraph::builder::makeStridedSlice(input, begin, end, strides, ngPrc, beginMask, endMask,
                                                std::vector<int64_t>(inputShape.size(), 0),
                                                std::vector<int64_t>(inputShape.size(), 0),
                                                std::vector<int64_t>(inputShape.size(), 0));

    ngraph::Shape const_shape(inputShape.size(), 1);
    const_shape.back() = 32;
    auto const_input = ngraph::builder::makeConstant(ngPrc, const_shape, std::vector<float>{}, true);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{const_input, ss}, inputShape.size() - 1);

    function = std::make_shared<ngraph::Function>(concat, params, "StridedSliceConcatTest");
}

}  // namespace SubgraphTestsDefinitions
