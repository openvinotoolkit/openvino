// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/bucketize.hpp"

namespace LayerTestsDefinitions {

    std::string BucketizeLayerTest::getTestCaseName(testing::TestParamInfo<bucketizeParamsTuple> obj) {
        InferenceEngine::SizeVector dataShape;
        InferenceEngine::SizeVector bucketsShape;
        bool with_right_bound;
        InferenceEngine::Precision inPrc;
        InferenceEngine::Precision netPrc;
        std::string targetDevice;

        std::tie(dataShape, bucketsShape, with_right_bound, inPrc, netPrc, targetDevice) = obj.param;

        std::ostringstream result;
        result << "DS=" << CommonTestUtils::vec2str(dataShape) << "_";
        result << "BS=" << CommonTestUtils::vec2str(bucketsShape) << "_";
        if (with_right_bound)
            result << "rightIntervalEdge_";
        else
            result << "leftIntervalEdge_";
        result << "inPrc=" << inPrc.name() << "_";
        result << "netPrc=" << netPrc.name() << "_";
        result << "trgDev=" << targetDevice;
        return result.str();
    }

    void BucketizeLayerTest::SetUp() {
        InferenceEngine::SizeVector dataShape;
        InferenceEngine::SizeVector bucketsShape;
        bool with_right_bound;
        InferenceEngine::Precision inPrc;
        InferenceEngine::Precision netPrc;

        std::tie(dataShape, bucketsShape, with_right_bound, inPrc, netPrc, targetDevice) = this->GetParam();

        auto ngInPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        auto ngNetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrc);
        auto params = ngraph::builder::makeParams(ngInPrc, {dataShape, bucketsShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto bucketize = std::make_shared<ngraph::op::v3::Bucketize>(params[0], params[1], ngNetPrc, with_right_bound);
        function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(bucketize), params, "Bucketize");
    }
} // namespace LayerTestsDefinitions
