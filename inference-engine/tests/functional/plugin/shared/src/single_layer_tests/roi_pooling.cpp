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

    std::string ROIPoolingLayerTest::getTestCaseName(testing::TestParamInfo<roiPoolingParamsTuple> obj) {
        std::vector<size_t> inputShape;
        std::vector<size_t> coordsShape;
        std::vector<size_t> poolShape;
        float spatialScale;
        ngraph::op::ROIPooling::ROIPoolingMethod poolMethod;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(inputShape, coordsShape, poolShape, spatialScale, poolMethod, netPrecision, targetDevice) = obj.param;

        std::ostringstream result;

        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "CS=" << CommonTestUtils::vec2str(coordsShape) << "_";
        result << "PS=" << CommonTestUtils::vec2str(poolShape) << "_";
        result << "Scale=" << spatialScale << "_";
        switch (poolMethod) {
            case ngraph::op::ROIPooling::ROIPoolingMethod::Max:
                result << "Max_";
                break;
            case ngraph::op::ROIPooling::ROIPoolingMethod::Bilinear:
                result << "Bilinear_";
                break;
        }

        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetDevice;
        return result.str();
    }

    void ROIPoolingLayerTest::SetUp() {
        InferenceEngine::SizeVector inputShape;
        InferenceEngine::SizeVector coordsShape;
        std::vector<size_t> poolShape;
        float spatialScale;
        ngraph::op::ROIPooling::ROIPoolingMethod poolMethod;
        InferenceEngine::Precision netPrecision;

        std::tie(inputShape, coordsShape, poolShape, spatialScale, poolMethod, netPrecision, targetDevice) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape, coordsShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto roi_pooling = std::make_shared<ngraph::op::ROIPooling>(paramOuts[0], paramOuts[1], poolShape, spatialScale, poolMethod);
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(roi_pooling)};
        function = std::make_shared<ngraph::Function>(results, params, "roi_pooling");
    }

    TEST_P(ROIPoolingLayerTest, CompareWithRefs) {
        Run();
    }
}  // namespace LayerTestsDefinitions
