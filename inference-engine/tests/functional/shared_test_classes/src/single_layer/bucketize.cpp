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

    InferenceEngine::Blob::Ptr BucketizeLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
        InferenceEngine::Blob::Ptr blobPtr;
        const std::string name = info.name();
        std::random_device rd{};
        if (name == "a_data") {
            auto data_shape = info.getTensorDesc().getDims();
            auto data_size = std::accumulate(begin(data_shape), end(data_shape), 1, std::multiplies<uint64_t>());
            blobPtr = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), data_size * 5, 0, 10, rd());
        } else if (name == "b_buckets") {
            blobPtr = FuncTestUtils::createAndFillBlobUniqueSequence(info.getTensorDesc(), 0, 10, rd());
        }
        return blobPtr;
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
        auto params = ngraph::builder::makeParams(ngInPrc, {{"a_data", dataShape}, {"b_buckets", bucketsShape}});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto bucketize = std::make_shared<ngraph::op::v3::Bucketize>(paramOuts[0], paramOuts[1], ngNetPrc, with_right_bound);
        function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(bucketize), params, "Bucketize");
    }
} // namespace LayerTestsDefinitions
