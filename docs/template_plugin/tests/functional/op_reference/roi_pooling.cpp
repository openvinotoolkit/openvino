// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <tuple>

#include "base_reference_test.hpp"

using namespace ngraph;
using namespace InferenceEngine;

struct ROIPoolingParams {
    ROIPoolingParams(const size_t iH, const size_t iW, const size_t ch, const size_t rois,
                     const size_t oH, const size_t oW, const float sS, const std::string mode,
                     void inputInit(float&),
                     const std::vector<float>& proposalValues, const std::vector<float>& outputValues,
                     size_t iSize = 0, size_t pSize = 0, size_t oSize = 0)
        : inputH(iH), inputW(iW), channelCount(ch), roiCount(rois), outputH(oH), outputW(oW), spatialScale(sS), poolingMode(mode),
        proposal(CreateBlob(element::f32, proposalValues, pSize)), refData(CreateBlob(element::f32, outputValues, oSize)) {
        auto inputSize = iH * iW * ch;
        std::vector<float> inputValues(inputSize);
        std::for_each(inputValues.begin(), inputValues.end(), inputInit);
        featureMap = CreateBlob(element::f32, inputValues, inputSize);
    }
    size_t inputH;
    size_t inputW;
    size_t channelCount;
    size_t roiCount;
    size_t outputH;
    size_t outputW;
    float spatialScale;
    std::string poolingMode;
    InferenceEngine::Blob::Ptr featureMap;
    InferenceEngine::Blob::Ptr proposal;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceRoiPoolingLayerTest : public testing::TestWithParam<ROIPoolingParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.inputH, params.inputW, params.channelCount, params.roiCount,
                                  params.outputH, params.outputW, params.spatialScale, params.poolingMode);
        inputData = {params.featureMap, params.proposal};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ROIPoolingParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "IS=" << param.inputH << "," << param.inputW << "_";
        result << "OS=" << param.outputH << "," << param.outputW << "_";
        result << "Ch=" << param.channelCount << "_";
        result << "Rois=" << param.roiCount << "_";
        result << "Ss=" << param.spatialScale << "_";
        result << "Mode=" << param.poolingMode << "_";
        result << std::to_string(obj.index);
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const size_t i_h, const size_t i_w, const size_t ch, const size_t roi_count,
                                                    const size_t o_h, const size_t o_w, const float spat_scale, const std::string mode) {
        Shape feat_map_shape{1, ch, i_h, i_w};
        Shape rois_shape{roi_count, 5};
        Shape pooled_shape{o_h, o_w};
        Shape output_shape{roi_count, ch, o_h, o_w};

        const auto feat_map = std::make_shared<op::Parameter>(element::f32, feat_map_shape);
        const auto rois = std::make_shared<op::Parameter>(element::f32, rois_shape);
        const auto roi_pooling = std::make_shared<op::v0::ROIPooling>(feat_map, rois, pooled_shape, spat_scale, mode);
        return std::make_shared<Function>(roi_pooling, ParameterVector{feat_map, rois});
    }
};

TEST_P(ReferenceRoiPoolingLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_ROIPooling_With_Hardcoded_Refs, ReferenceRoiPoolingLayerTest,
    ::testing::Values(
            // roi_pooling_1x1_max
            ROIPoolingParams(6, 6,              // iH, iW
                             3, 3,              // channels, rois
                             1, 1,              // oH, oW
                             1.f, "max",        // scale, mode
                             [] (float &x) {static float n = 0; x = 1.f * n++ / 10;},
                             std::vector<float> {0, 1, 1, 2, 3, 0, 1, 1, 2, 3, 0, 1, 1, 2, 3},
                             std::vector<float> {2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f}),
            // roi_pooling_2x2_max
            ROIPoolingParams(6, 6,              // iH, iW
                             1, 3,              // channels, rois
                             2, 2,              // oH, oW
                             1.f, "max",        // scale, mode
                             [] (float &x) {static float n = 0; x = 1.f * n++ / 10;},
                             std::vector<float> {0, 1, 1, 3, 3, 0, 1, 2, 2, 4, 0, 0, 1, 4, 5},
                             std::vector<float> {1.4f, 1.5f, 2.0f, 2.1f, 1.9f, 2.0f, 2.5f, 2.6f, 2.0f, 2.2f, 3.2f, 3.4f}),
            // roi_pooling_1x1_bilinear
            ROIPoolingParams(6, 6,              // iH, iW
                             3, 2,              // channels, rois
                             1, 1,              // oH, oW
                             1.f, "bilinear",   // scale, mode
                             [] (float &x) {static float n = 0; x = 1.f * n++ / 10;},
                             std::vector<float> {0, 0.2, 0.2, 0.4, 0.4, 0, 0.2, 0.2, 0.6, 0.6},
                             std::vector<float> {1.05f, 4.65f, 8.25f, 1.4f, 5.0f, 8.6f}),
            // roi_pooling_2x2_bilinear
            ROIPoolingParams(8, 8,              // iH, iW
                             1, 3,              // channels, rois
                             2, 2,              // oH, oW
                             1.f, "bilinear",   // scale, mode
                             [] (float &x) {static float n = 0; x = 1.f * n++ / 10;},
                             std::vector<float> {0.f, 0.15f, 0.2f, 0.75f, 0.8f,
                                                 0.f, 0.15f, 0.2f, 0.75f, 0.8f,
                                                 0.f, 0.15f, 0.2f, 0.75f, 0.8f},
                             std::vector<float> {1.225f, 1.645f, 4.585f, 5.005f,
                                                 1.225f, 1.645f, 4.585f, 5.005f,
                                                 1.225f, 1.645f, 4.585f, 5.005f}),
            // roi_pooling_2x2_bilinear_border_proposal
            ROIPoolingParams(50, 50,            // iH, iW
                             1, 1,              // channels, rois
                             4, 4,              // oH, oW
                             1.f, "bilinear",   // scale, mode
                             [] (float &x) {x = 1.f;},
                             std::vector<float> {0.f, 0.f, 0.248046786f, 0.471333951f, 1.f},
                             std::vector<float>(16, 1.f))),
                             ReferenceRoiPoolingLayerTest::getTestCaseName);
