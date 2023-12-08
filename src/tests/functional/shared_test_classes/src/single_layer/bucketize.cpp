// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/bucketize.hpp"

namespace LayerTestsDefinitions {
    std::string BucketizeLayerTest::getTestCaseName(const testing::TestParamInfo<bucketizeParamsTuple>& obj) {
        InferenceEngine::SizeVector dataShape;
        InferenceEngine::SizeVector bucketsShape;
        bool with_right_bound;
        InferenceEngine::Precision inDataPrc;
        InferenceEngine::Precision inBucketsPrc;
        InferenceEngine::Precision netPrc;
        std::string targetDevice;

        std::tie(dataShape, bucketsShape, with_right_bound, inDataPrc, inBucketsPrc, netPrc, targetDevice) = obj.param;

        std::ostringstream result;
        result << "DS=" << ov::test::utils::vec2str(dataShape) << "_";
        result << "BS=" << ov::test::utils::vec2str(bucketsShape) << "_";
        if (with_right_bound)
            result << "rightIntervalEdge_";
        else
            result << "leftIntervalEdge_";
        result << "inDataPrc=" << inDataPrc.name() << "_";
        result << "inBucketsPrc=" << inBucketsPrc.name() << "_";
        result << "netPrc=" << netPrc.name() << "_";
        result << "trgDev=" << targetDevice;
        return result.str();
    }

    InferenceEngine::Blob::Ptr BucketizeLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
        InferenceEngine::Blob::Ptr blobPtr;
        const std::string name = info.name();
        if (name == "a_data") {
            auto data_shape = info.getTensorDesc().getDims();
            auto data_size = std::accumulate(begin(data_shape), end(data_shape), 1, std::multiplies<uint64_t>());
            blobPtr = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), data_size * 5, 0, 10, 7235346);
        } else if (name == "b_buckets") {
            blobPtr = FuncTestUtils::createAndFillBlobUniqueSequence(info.getTensorDesc(), 0, 10, 8234231);
        }
        return blobPtr;
    }

    void BucketizeLayerTest::SetUp() {
        InferenceEngine::SizeVector dataShape;
        InferenceEngine::SizeVector bucketsShape;
        bool with_right_bound;
        InferenceEngine::Precision inDataPrc;
        InferenceEngine::Precision inBucketsPrc;
        InferenceEngine::Precision netPrc;

        std::tie(dataShape, bucketsShape, with_right_bound, inDataPrc, inBucketsPrc, netPrc, targetDevice) = this->GetParam();

        auto ngInDataPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inDataPrc);
        auto ngInBucketsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inBucketsPrc);
        auto ngNetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrc);
        auto data = std::make_shared<ngraph::op::Parameter>(ngInDataPrc, ngraph::Shape(dataShape));
        data->set_friendly_name("a_data");
        auto buckets = std::make_shared<ngraph::op::Parameter>(ngInBucketsPrc, ngraph::Shape(bucketsShape));
        buckets->set_friendly_name("b_buckets");
        auto bucketize = std::make_shared<ngraph::op::v3::Bucketize>(data, buckets, ngNetPrc, with_right_bound);
        function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(bucketize), ngraph::ParameterVector{data, buckets}, "Bucketize");
    }
} // namespace LayerTestsDefinitions
