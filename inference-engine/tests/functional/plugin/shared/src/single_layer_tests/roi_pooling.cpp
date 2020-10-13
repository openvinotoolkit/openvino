// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/layer_test_utils.hpp"


#include "single_layer_tests/roi_pooling.hpp"

namespace LayerTestsDefinitions {

    std::string ROIPoolingLayerTest::getTestCaseName(testing::TestParamInfo<roiPoolingLayerTestParamsSet> obj) {
        roiPoolingSpecificParams roiPoolParams;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        std::vector<size_t> inputShape;
        std::vector<size_t> coordsShape;
        std::string targetDevice;
        std::tie(roiPoolParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, coordsShape, targetDevice) = obj.param;
        ngraph::helpers::ROIPoolingTypes poolType;
        std::vector<size_t> poolShape;
        float scale;
        std::tie(poolType, poolShape, scale) = roiPoolParams;

        std::ostringstream result;

        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "CS=" << CommonTestUtils::vec2str(coordsShape) << "_";
        switch (poolType) {
            case ngraph::helpers::ROIPoolingTypes::ROI_MAX:
                result << "Max_";
                break;
            case ngraph::helpers::ROIPoolingTypes::ROI_BILINEAR:
                result << "Bilinear_";
                break;
        }
        result << "PS" << CommonTestUtils::vec2str(poolShape) << "_";
        result << "S" << scale << "_";

        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "inL=" << inLayout << "_";
        result << "outL=" << outLayout << "_";
        result << "trgDev=" << targetDevice;
        return result.str();
    }

    void ROIPoolingLayerTest::SetUp() {
        roiPoolingSpecificParams roiPoolParams;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        InferenceEngine::SizeVector inputShape;
        InferenceEngine::SizeVector coordsShape;
        std::string targetDevice;
        std::tie(roiPoolParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, coordsShape, targetDevice) = this->GetParam();
        ngraph::helpers::ROIPoolingTypes poolType;
        std::vector<size_t> poolShape;
        float scale;
        std::tie(poolType, poolShape, scale) = roiPoolParams;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape, coordsShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::shared_ptr<ngraph::Node> roi_pooling = ngraph::builder::makeROIPooling(paramOuts[0],
                                                                                    paramOuts[1],
                                                                                    poolShape,
                                                                                    scale,
                                                                                    poolType);

        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(roi_pooling)};
        function = std::make_shared<ngraph::Function>(results, params, "roi_pooling");
    }

    TEST_P(ROIPoolingLayerTest, CompareWithRefs) {
        Run();
    }
}  // namespace LayerTestsDefinitions
