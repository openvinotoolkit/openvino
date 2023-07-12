// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/grid_sample.hpp"

namespace LayerTestsDefinitions {

std::string GridSampleLayerTest::getTestCaseName(const testing::TestParamInfo<GridSampleParams>& obj) {
    InferenceEngine::SizeVector dataShape;
    InferenceEngine::SizeVector gridShape;
    decltype(ngraph::op::v9::GridSample::Attributes::align_corners) alignCorners;
    decltype(ngraph::op::v9::GridSample::Attributes::mode) mode;
    decltype(ngraph::op::v9::GridSample::Attributes::padding_mode) paddingMode;
    InferenceEngine::Precision inDataPrc;
    InferenceEngine::Precision inGridPrc;
    std::string targetDevice;

    std::tie(dataShape, gridShape, alignCorners, mode, paddingMode, inDataPrc, inGridPrc, targetDevice) = obj.param;

    std::ostringstream result;
    result << "DS=" << ov::test::utils::vec2str(dataShape) << "_";
    result << "GS=" << ov::test::utils::vec2str(gridShape) << "_";
    result << "AlignCorners=" << alignCorners << "_";
    result << "Mode=" << ov::as_string(mode) << "_";
    result << "PaddingMode=" << ov::as_string(paddingMode) << "_";
    result << "inDataPrc=" << inDataPrc.name() << "_";
    result << "inGridPrc=" << inGridPrc.name() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void GridSampleLayerTest::SetUp() {
    InferenceEngine::SizeVector dataShape;
    InferenceEngine::SizeVector gridShape;
    decltype(ngraph::op::v9::GridSample::Attributes::align_corners) alignCorners;
    decltype(ngraph::op::v9::GridSample::Attributes::mode) mode;
    decltype(ngraph::op::v9::GridSample::Attributes::padding_mode) paddingMode;
    InferenceEngine::Precision inDataPrc;
    InferenceEngine::Precision inGridPrc;

    std::tie(dataShape, gridShape, alignCorners, mode, paddingMode, inDataPrc, inGridPrc, targetDevice) =
        this->GetParam();

    auto ngInDataPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inDataPrc);
    auto ngInGridPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inGridPrc);
    auto data = std::make_shared<ngraph::op::Parameter>(ngInDataPrc, ngraph::Shape(dataShape));
    auto grid = std::make_shared<ngraph::op::Parameter>(ngInGridPrc, ngraph::Shape(gridShape));
    auto gridSample = std::make_shared<ngraph::op::v9::GridSample>(
        data,
        grid,
        ngraph::op::v9::GridSample::Attributes(alignCorners, mode, paddingMode));
    function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(gridSample),
                                                  ngraph::ParameterVector{data, grid},
                                                  "GridSample");
}

}  // namespace LayerTestsDefinitions
