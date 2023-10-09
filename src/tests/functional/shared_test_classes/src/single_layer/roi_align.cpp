// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/roi_align.hpp"

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset9.hpp>

#include "ov_models/builders.hpp"
#include "openvino/core/enum_names.hpp"

using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

namespace LayerTestsDefinitions {

std::string ROIAlignLayerTest::getTestCaseName(const testing::TestParamInfo<roialignParams>& obj) {
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

    result << "in_shape=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "coord_shape=" << ov::test::utils::vec2str(coordsShape) << "_";
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

void ROIAlignLayerTest::fillCoordTensor(std::vector<float>& coords, int height, int width,
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
void ROIAlignLayerTest::fillIdxTensor(std::vector<int>& idx, int batchSize) {
    int batchId = 0;
    for (int i = 0; i < idx.size(); i++) {
        idx[i] = batchId;
        batchId = (batchId + 1) % batchSize;
    }
}

void ROIAlignLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<size_t> coordsShape;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape,
             coordsShape,
             pooledH,
             pooledW,
             spatialScale,
             poolingRatio,
             poolingMode,
             netPrecision,
             targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    std::vector<float> proposalVector;
    std::vector<int> roiIdxVector;
    proposalVector.resize(coordsShape[0] * 4);
    roiIdxVector.resize(coordsShape[0]);

    fillCoordTensor(proposalVector, inputShape[2], inputShape[3], spatialScale, poolingRatio, pooledH, pooledW);
    fillIdxTensor(roiIdxVector, inputShape[0]);
    ngraph::Shape idxShape = {coordsShape[0]};

    auto coords = std::make_shared<ngraph::opset1::Constant>(ngPrc, coordsShape, proposalVector.data());
    auto roisIdx = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i32, idxShape, roiIdxVector.data());

    std::shared_ptr<ngraph::Node> roiAlign = std::make_shared<ngraph::opset3::ROIAlign>(paramOuts[0],
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

std::string ROIAlignV9LayerTest::getTestCaseName(const testing::TestParamInfo<roialignV9Params>& obj) {
    std::vector<size_t> inputShape;
    std::vector<size_t> coordsShape;

    int pooledH;
    int pooledW;
    float spatialScale;
    int poolingRatio;
    std::string poolingMode;
    std::string roiAlignedMode;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(inputShape,
             coordsShape,
             pooledH,
             pooledW,
             spatialScale,
             poolingRatio,
             poolingMode,
             roiAlignedMode,
             netPrecision,
             targetDevice) = obj.param;

    std::ostringstream result;

    result << "in_shape=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "coord_shape=" << ov::test::utils::vec2str(coordsShape) << "_";
    result << "pooled_h=" << pooledH << "_";
    result << "pooled_w=" << pooledW << "_";
    result << "spatial_scale=" << spatialScale << "_";
    result << "pooling_ratio=" << poolingRatio << "_";
    result << "mode=" << poolingMode << "_";
    result << "mode=" << roiAlignedMode << "_";
    result << "prec=" << netPrecision.name() << "_";
    result << "dev=" << targetDevice;
    return result.str();
}

void ROIAlignV9LayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<size_t> coordsShape;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape,
             coordsShape,
             pooledH,
             pooledW,
             spatialScale,
             poolingRatio,
             poolingMode,
             roiAlignedMode,
             netPrecision,
             targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    std::vector<float> proposalVector;
    std::vector<int> roiIdxVector;
    proposalVector.resize(coordsShape[0] * 4);
    roiIdxVector.resize(coordsShape[0]);

    ROIAlignLayerTest::fillCoordTensor(proposalVector,
                                       inputShape[2],
                                       inputShape[3],
                                       spatialScale,
                                       poolingRatio,
                                       pooledH,
                                       pooledW);
    ROIAlignLayerTest::fillIdxTensor(roiIdxVector, inputShape[0]);
    ngraph::Shape idxShape = {coordsShape[0]};

    auto coords = std::make_shared<ngraph::opset1::Constant>(ngPrc, coordsShape, proposalVector.data());
    auto roisIdx = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i32, idxShape, roiIdxVector.data());

    std::shared_ptr<ngraph::Node> roiAlign = std::make_shared<ngraph::opset9::ROIAlign>(
            paramOuts[0],
            coords,
            roisIdx,
            pooledH,
            pooledW,
            poolingRatio,
            spatialScale,
            ov::EnumNames<ngraph::opset9::ROIAlign::PoolingMode>::as_enum(poolingMode),
            ov::EnumNames<ngraph::opset9::ROIAlign::AlignedMode>::as_enum(roiAlignedMode));

    ngraph::ResultVector results{std::make_shared<ngraph::opset9::Result>(roiAlign)};
    function = std::make_shared<ngraph::Function>(results, params, "roi_align");
}
}  // namespace LayerTestsDefinitions
