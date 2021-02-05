// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/single_layer/roi_align.hpp"

using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

namespace LayerTestsDefinitions {

std::string ROIAlignLayerTest::getTestCaseName(testing::TestParamInfo<roialignParams> obj) {
    std::vector<size_t> inputShape;
    std::vector<size_t> coordsShape;

    int pooledH;
    int pooledW;
    float spatialScale;
    int poolingRatio;
    std::string poolingMode;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(inputShape, coordsShape, pooledH, pooledW, spatialScale, poolingRatio, poolingMode, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;

    result << "in_shape=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "coord_shape=" << CommonTestUtils::vec2str(coordsShape) << "_";
    result << "pooled_h=" << pooledH << "_";
    result << "pooled_w=" << pooledW << "_";
    result << "spatial_scale=" << spatialScale << "_";
    result << "pooling_ratio=" << poolingRatio << "_";
    result << "mode=" << poolingMode << "_";
    result << "prec=" << netPrecision.name() << "_";
    result << "dev=" << targetDevice;
    return result.str();
}

static int randInt(int low, int high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(low, high);
    return dis(gen);
}

static void fillCoordTensor(std::vector<float> coords, int height, int width,
                            float spatialScale, int pooledRatio, int pooledH, int pooledW) {
    int minRoiWidth = pooledW;
    int maxRoiWidth = width / pooledRatio;
    int minRoiHeight = pooledH;
    int maxRoiHeight = height / pooledRatio;

    for (int i = 0; i < coords.size() / 4; i++) {
        int sizeX = std::min(width, randInt(minRoiWidth, maxRoiWidth));
        int sizeY = std::min(height, randInt(minRoiHeight, maxRoiHeight));
        int startX = randInt(0, std::max(1, width - sizeX - 1));
        int startY = randInt(0, std::max(1, height - sizeY - 1));

        coords[i * 4] = startX / spatialScale;
        coords[i * 4 + 1] = startY / spatialScale;
        coords[i * 4 + 2] = (startX + sizeX - 1) / spatialScale;
        coords[i * 4 + 3] = (startY + sizeY - 1) / spatialScale;
    }
}
static void fillIdxTensor(std::vector<int> idx, int batchSize) {
    int batchId = 0;
    for (int i = 0; i < idx.size(); i++) {
        idx[i] = batchId;
        batchId = (batchId + 1) % batchSize;
    }
}

void ROIAlignLayerTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();
    auto inputShape = cnnNetwork.getInputShapes().begin()->second;
    for (const auto &input : cnnNetwork.getInputsInfo()) {
        const auto &info = input.second;
        Blob::Ptr blob;
        blob = GenerateInput(*info);
        inferRequest.SetBlob(info->name(), blob);
        inputs.push_back(blob);
    }
    inferRequest.Infer();
}

void ROIAlignLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<size_t> coordsShape;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, coordsShape, pooledH, pooledW,
             spatialScale, poolingRatio, poolingMode, netPrecision, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    std::vector<float> proposalVector;
    std::vector<int> roiIdxVector;
    proposalVector.resize(coordsShape[0] * 4);
    roiIdxVector.resize(coordsShape[0]);

    fillCoordTensor(proposalVector, inputShape[2], inputShape[3],
            spatialScale, poolingRatio, pooledH, pooledW);
    fillIdxTensor(roiIdxVector, inputShape[0]);
    ngraph::Shape idxShape = { coordsShape[0] };

    auto coords = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, coordsShape, proposalVector.data());
    auto roisIdx = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i32, idxShape, roiIdxVector.data());

    std::shared_ptr<ngraph::Node> roiAlign =
            std::make_shared<ngraph::opset3::ROIAlign>(paramOuts[0],
                                                    coords,
                                                    roisIdx,
                                                    pooledH,
                                                    pooledW,
                                                    poolingRatio,
                                                    spatialScale,
                                                    poolingMode);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(roiAlign)};
    function = std::make_shared<ngraph::Function>(results, params, "roi_align");
}
}  // namespace LayerTestsDefinitions
