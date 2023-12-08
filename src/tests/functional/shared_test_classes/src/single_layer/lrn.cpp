// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/lrn.hpp"

namespace LayerTestsDefinitions {

std::string LrnLayerTest::getTestCaseName(const testing::TestParamInfo<lrnLayerTestParamsSet>& obj) {
    double alpha, beta, bias;
    size_t size;
    std::vector<int64_t> axes;
    InferenceEngine::Precision  netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    std::vector<size_t> inputShapes;
    std::string targetDevice;
    std::tie(alpha, beta, bias, size, axes, netPrecision, inPrc, outPrc, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << separator;
    result << "Alpha=" << alpha << separator;
    result << "Beta=" << beta << separator;
    result << "Bias=" << bias << separator;
    result << "Size=" << size << separator;
    result << "Axes=" << ov::test::utils::vec2str(axes) << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "inPRC=" << inPrc.name() << separator;
    result << "outPRC=" << outPrc.name() << separator;
    result << "trgDev=" << targetDevice;

    return result.str();
}

void LrnLayerTest::SetUp() {
    std::vector<size_t> inputShapes;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    double alpha, beta, bias;
    size_t size;
    std::vector<int64_t> axes;
    std::tie(alpha, beta, bias, size, axes, netPrecision, inPrc, outPrc, inputShapes, targetDevice) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes))};

    auto axes_node = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{axes.size()}, axes.data());
    auto lrn = std::make_shared<ngraph::opset3::LRN>(params[0], axes_node, alpha, beta, bias, size);
    ngraph::ResultVector results {std::make_shared<ngraph::opset3::Result>(lrn)};
    function = std::make_shared<ngraph::Function>(results, params, "lrn");
}
}  // namespace LayerTestsDefinitions
