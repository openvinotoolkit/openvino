// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/shape_of.hpp"

namespace LayerTestsDefinitions {

    std::string ShapeOfLayerTest::getTestCaseName(testing::TestParamInfo<shapeOfParams> obj) {
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision inputPrecision;
        InferenceEngine::Precision outputPrecision;
        std::string targetDevice;
        std::tie(inputPrecision, outputPrecision, inputShapes, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
        result << "Precision=" << inputPrecision.name() << "_";
        result << "Output Precision=" << outputPrecision.name() << "_";
        result << "TargetDevice=" << targetDevice;
        return result.str();
    }

    void ShapeOfLayerTest::SetUp() {
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision inputPrecision;
        std::tie(inputPrecision, outPrc, inputShapes, targetDevice) = this->GetParam();
        auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
        auto outType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);
        ov::ParameterVector param {std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(inputShapes))};
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(param));
        auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(paramOuts[0], outType);
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(shapeOf)};
        function = std::make_shared<ngraph::Function>(results, param, "shapeOf");
    }

}  // namespace LayerTestsDefinitions
