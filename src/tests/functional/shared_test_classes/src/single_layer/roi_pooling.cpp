// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/roi_pooling.hpp"

namespace LayerTestsDefinitions {

    std::string ROIPoolingLayerTest::getTestCaseName(const testing::TestParamInfo<roiPoolingParamsTuple>& obj) {
        std::vector<size_t> inputShape;
        std::vector<size_t> coordsShape;
        std::vector<size_t> poolShape;
        float spatial_scale;
        ngraph::helpers::ROIPoolingTypes pool_method;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(inputShape, coordsShape, poolShape, spatial_scale, pool_method, netPrecision, targetDevice) = obj.param;

        std::ostringstream result;

        result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
        result << "CS=" << ov::test::utils::vec2str(coordsShape) << "_";
        result << "PS=" << ov::test::utils::vec2str(poolShape) << "_";
        result << "Scale=" << spatial_scale << "_";
        switch (pool_method) {
            case ngraph::helpers::ROIPoolingTypes::ROI_MAX:
                result << "Max_";
                break;
            case ngraph::helpers::ROIPoolingTypes::ROI_BILINEAR:
                result << "Bilinear_";
                break;
        }
        result << "netPRC=" << netPrecision.name() << "_";
        result << "trgDev=" << targetDevice;
        return result.str();
    }

    void ROIPoolingLayerTest::GenerateInputs() {
        auto feat_map_shape = cnnNetwork.getInputShapes().begin()->second;

        const auto is_roi_max_mode = (pool_method == ngraph::helpers::ROIPoolingTypes::ROI_MAX);

        const int height = is_roi_max_mode ? feat_map_shape[2] / spatial_scale : 1;
        const int width  = is_roi_max_mode ? feat_map_shape[3] / spatial_scale : 1;

        size_t it = 0;
        for (const auto &input : cnnNetwork.getInputsInfo()) {
            const auto &info = input.second;
            InferenceEngine::Blob::Ptr blob;

            if (it == 1) {
                blob = make_blob_with_precision(info->getTensorDesc());
                blob->allocate();
                ov::test::utils::fill_data_roi<InferenceEngine::Precision::FP32>(blob, feat_map_shape[0] - 1,
                                                                                 height, width, 1.0f, is_roi_max_mode);
            } else {
                blob = GenerateInput(*info);
            }
            inputs.push_back(blob);
            it++;
        }
    }

    void ROIPoolingLayerTest::SetUp() {
        InferenceEngine::SizeVector inputShape;
        InferenceEngine::SizeVector coordsShape;
        InferenceEngine::SizeVector poolShape;
        InferenceEngine::Precision netPrecision;

        threshold = 0.08f;

        std::tie(inputShape, coordsShape, poolShape, spatial_scale, pool_method, netPrecision, targetDevice) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                                    std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(coordsShape))};
        std::shared_ptr<ov::Node> roi_pooling;
        if (ov::test::utils::ROIPoolingTypes::ROI_MAX == pool_method) {
            roi_pooling = std::make_shared<ov::op::v0::ROIPooling>(params[0], params[1], poolShape, spatial_scale, "max");
        } else {
            roi_pooling = std::make_shared<ov::op::v0::ROIPooling>(params[0], params[1], poolShape, spatial_scale, "bilinear");
        }
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(roi_pooling)};
        function = std::make_shared<ngraph::Function>(results, params, "roi_pooling");
    }
}  // namespace LayerTestsDefinitions
