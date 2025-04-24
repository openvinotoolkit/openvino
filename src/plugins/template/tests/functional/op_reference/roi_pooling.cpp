// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_pooling.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

struct ROIPoolingParams {
    template <class T>
    ROIPoolingParams(const size_t iH,
                     const size_t iW,
                     const size_t ch,
                     const size_t rois,
                     const size_t oH,
                     const size_t oW,
                     const float sS,
                     const std::string mode,
                     const ov::element::Type& type,
                     const std::vector<T>& inputValues,
                     const std::vector<T>& proposalValues,
                     const std::vector<T>& outputValues)
        : inputH(iH),
          inputW(iW),
          channelCount(ch),
          roiCount(rois),
          outputH(oH),
          outputW(oW),
          spatialScale(sS),
          poolingMode(mode),
          dataType(type),
          featureMap(CreateTensor(type, inputValues)),
          proposal(CreateTensor(type, proposalValues)),
          refData(CreateTensor(Shape{rois, ch, oH, oW}, type, outputValues)) {}
    size_t inputH;
    size_t inputW;
    size_t channelCount;
    size_t roiCount;
    size_t outputH;
    size_t outputW;
    float spatialScale;
    std::string poolingMode;
    ov::element::Type dataType;
    ov::Tensor featureMap;
    ov::Tensor proposal;
    ov::Tensor refData;

public:
    template <class T>
    inline static std::vector<T> increasinglyFilledBlob(size_t size) {
        std::vector<T> inputValues;
        T one = 1;
        for (size_t i = 0; i < size; i++) {
            inputValues.push_back(one * i / 10);
        }
        return inputValues;
    }
    template <class T>
    inline static std::vector<T> equallyFilledBlob(size_t size, T value) {
        std::vector<T> inputValues;
        for (size_t i = 0; i < size; i++) {
            inputValues.push_back(value);
        }
        return inputValues;
    }
};

class ReferenceRoiPoolingLayerTest : public testing::TestWithParam<ROIPoolingParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.inputH,
                                  params.inputW,
                                  params.channelCount,
                                  params.roiCount,
                                  params.outputH,
                                  params.outputW,
                                  params.spatialScale,
                                  params.poolingMode,
                                  params.dataType);
        inputData = {params.featureMap, params.proposal};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ROIPoolingParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "IS=" << param.inputH << "," << param.inputW << "_";
        result << "OS=" << param.outputH << "," << param.outputW << "_";
        result << "Ch=" << param.channelCount << "_";
        result << "Rois=" << param.roiCount << "_";
        result << "Ss=" << param.spatialScale << "_";
        result << "Mode=" << param.poolingMode << "_";
        result << "Prec=" << param.dataType << "_";
        result << std::to_string(obj.index);
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const size_t i_h,
                                                 const size_t i_w,
                                                 const size_t ch,
                                                 const size_t roi_count,
                                                 const size_t o_h,
                                                 const size_t o_w,
                                                 const float spat_scale,
                                                 const std::string mode,
                                                 const ov::element::Type& type) {
        Shape feat_map_shape{1, ch, i_h, i_w};
        Shape rois_shape{roi_count, 5};
        Shape pooled_shape{o_h, o_w};
        Shape output_shape{roi_count, ch, o_h, o_w};

        const auto feat_map = std::make_shared<op::v0::Parameter>(type, feat_map_shape);
        const auto rois = std::make_shared<op::v0::Parameter>(type, rois_shape);
        const auto roi_pooling = std::make_shared<op::v0::ROIPooling>(feat_map, rois, pooled_shape, spat_scale, mode);
        return std::make_shared<ov::Model>(roi_pooling, ParameterVector{feat_map, rois});
    }
};

TEST_P(ReferenceRoiPoolingLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_ROIPooling_With_Hardcoded_Refs,
    ReferenceRoiPoolingLayerTest,
    ::testing::Values(
        // fp32
        // roi_pooling_1x1_max
        ROIPoolingParams(6,
                         6,  // iH, iW
                         3,
                         3,  // channels, rois
                         1,
                         1,  // oH, oW
                         1.f,
                         "max",  // scale, mode
                         element::f32,
                         ROIPoolingParams::increasinglyFilledBlob<float>(3 * 6 * 6),
                         std::vector<float>{0, 1, 1, 2, 3, 0, 1, 1, 2, 3, 0, 1, 1, 2, 3},
                         std::vector<float>{2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f}),
        // roi_pooling_2x2_max
        ROIPoolingParams(6,
                         6,  // iH, iW
                         1,
                         3,  // channels, rois
                         2,
                         2,  // oH, oW
                         1.f,
                         "max",  // scale, mode
                         element::f32,
                         ROIPoolingParams::increasinglyFilledBlob<float>(1 * 6 * 6),
                         std::vector<float>{0, 1, 1, 3, 3, 0, 1, 2, 2, 4, 0, 0, 1, 4, 5},
                         std::vector<float>{1.4f, 1.5f, 2.0f, 2.1f, 1.9f, 2.0f, 2.5f, 2.6f, 2.0f, 2.2f, 3.2f, 3.4f}),
        // roi_pooling_1x1_bilinear
        ROIPoolingParams(6,
                         6,  // iH, iW
                         3,
                         2,  // channels, rois
                         1,
                         1,  // oH, oW
                         1.f,
                         "bilinear",  // scale, mode
                         element::f32,
                         ROIPoolingParams::increasinglyFilledBlob<float>(3 * 6 * 6),
                         std::vector<float>{0, 0.2, 0.2, 0.4, 0.4, 0, 0.2, 0.2, 0.6, 0.6},
                         std::vector<float>{1.05f, 4.65f, 8.25f, 1.4f, 5.0f, 8.6f}),
        // roi_pooling_2x2_bilinear
        ROIPoolingParams(
            8,
            8,  // iH, iW
            1,
            3,  // channels, rois
            2,
            2,  // oH, oW
            1.f,
            "bilinear",  // scale, mode
            element::f32,
            ROIPoolingParams::increasinglyFilledBlob<float>(1 * 8 * 8),
            std::vector<
                float>{0.f, 0.15f, 0.2f, 0.75f, 0.8f, 0.f, 0.15f, 0.2f, 0.75f, 0.8f, 0.f, 0.15f, 0.2f, 0.75f, 0.8f},
            std::vector<
                float>{1.225f, 1.645f, 4.585f, 5.005f, 1.225f, 1.645f, 4.585f, 5.005f, 1.225f, 1.645f, 4.585f, 5.005f}),
        // roi_pooling_2x2_bilinear_border_proposal
        ROIPoolingParams(50,
                         50,  // iH, iW
                         1,
                         1,  // channels, rois
                         4,
                         4,  // oH, oW
                         1.f,
                         "bilinear",  // scale, mode
                         element::f32,
                         ROIPoolingParams::equallyFilledBlob<float>(1 * 50 * 50, 1),
                         std::vector<float>{0.f, 0.f, 0.248046786f, 0.471333951f, 1.f},
                         std::vector<float>(16, 1.f)),

        // bf16
        // roi_pooling_1x1_max
        ROIPoolingParams(6,
                         6,  // iH, iW
                         3,
                         3,  // channels, rois
                         1,
                         1,  // oH, oW
                         1.f,
                         "max",  // scale, mode
                         element::bf16,
                         ROIPoolingParams::increasinglyFilledBlob<bfloat16>(3 * 6 * 6),
                         std::vector<bfloat16>{0, 1, 1, 2, 3, 0, 1, 1, 2, 3, 0, 1, 1, 2, 3},
                         std::vector<bfloat16>{2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f}),
        // roi_pooling_2x2_max
        ROIPoolingParams(6,
                         6,  // iH, iW
                         1,
                         3,  // channels, rois
                         2,
                         2,  // oH, oW
                         1.f,
                         "max",  // scale, mode
                         element::bf16,
                         ROIPoolingParams::increasinglyFilledBlob<bfloat16>(1 * 6 * 6),
                         std::vector<bfloat16>{0, 1, 1, 3, 3, 0, 1, 2, 2, 4, 0, 0, 1, 4, 5},
                         std::vector<bfloat16>{1.4f, 1.5f, 2.0f, 2.1f, 1.9f, 2.0f, 2.5f, 2.6f, 2.0f, 2.2f, 3.2f, 3.4f}),
        // roi_pooling_1x1_bilinear
        ROIPoolingParams(6,
                         6,  // iH, iW
                         3,
                         2,  // channels, rois
                         1,
                         1,  // oH, oW
                         1.f,
                         "bilinear",  // scale, mode
                         element::bf16,
                         ROIPoolingParams::increasinglyFilledBlob<bfloat16>(3 * 6 * 6),
                         std::vector<bfloat16>{0, 0.2, 0.2, 0.4, 0.4, 0, 0.2, 0.2, 0.6, 0.6},
                         std::vector<bfloat16>{1.05f, 4.65f, 8.25f, 1.4f, 5.0f, 8.6f}),
        // roi_pooling_2x2_bilinear
        ROIPoolingParams(
            8,
            8,  // iH, iW
            1,
            3,  // channels, rois
            2,
            2,  // oH, oW
            1.f,
            "bilinear",  // scale, mode
            element::bf16,
            ROIPoolingParams::increasinglyFilledBlob<bfloat16>(1 * 8 * 8),
            std::vector<
                bfloat16>{0.f, 0.15f, 0.2f, 0.75f, 0.8f, 0.f, 0.15f, 0.2f, 0.75f, 0.8f, 0.f, 0.15f, 0.2f, 0.75f, 0.8f},
            std::vector<bfloat16>{1.225f,
                                  1.645f,
                                  4.585f,
                                  4.937f,
                                  1.225f,
                                  1.645f,
                                  4.585f,
                                  4.937f,
                                  1.225f,
                                  1.645f,
                                  4.585f,
                                  4.937f}),
        // fp16
        // roi_pooling_1x1_max
        ROIPoolingParams(6,
                         6,  // iH, iW
                         3,
                         3,  // channels, rois
                         1,
                         1,  // oH, oW
                         1.f,
                         "max",  // scale, mode
                         element::f16,
                         ROIPoolingParams::increasinglyFilledBlob<float16>(3 * 6 * 6),
                         std::vector<float16>{0, 1, 1, 2, 3, 0, 1, 1, 2, 3, 0, 1, 1, 2, 3},
                         std::vector<float16>{2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f}),
        // roi_pooling_2x2_max
        ROIPoolingParams(6,
                         6,  // iH, iW
                         1,
                         3,  // channels, rois
                         2,
                         2,  // oH, oW
                         1.f,
                         "max",  // scale, mode
                         element::f16,
                         ROIPoolingParams::increasinglyFilledBlob<float16>(1 * 6 * 6),
                         std::vector<float16>{0, 1, 1, 3, 3, 0, 1, 2, 2, 4, 0, 0, 1, 4, 5},
                         std::vector<float16>{1.4f, 1.5f, 2.0f, 2.1f, 1.9f, 2.0f, 2.5f, 2.6f, 2.0f, 2.2f, 3.2f, 3.4f}),
        // roi_pooling_1x1_bilinear
        ROIPoolingParams(6,
                         6,  // iH, iW
                         3,
                         2,  // channels, rois
                         1,
                         1,  // oH, oW
                         1.f,
                         "bilinear",  // scale, mode
                         element::f16,
                         ROIPoolingParams::increasinglyFilledBlob<float16>(3 * 6 * 6),
                         std::vector<float16>{0, 0.2, 0.2, 0.4, 0.4, 0, 0.2, 0.2, 0.6, 0.6},
                         std::vector<float16>{1.05f, 4.65f, 8.25f, 1.4f, 5.0f, 8.6f}),
        // roi_pooling_2x2_bilinear
        ROIPoolingParams(
            8,
            8,  // iH, iW
            1,
            3,  // channels, rois
            2,
            2,  // oH, oW
            1.f,
            "bilinear",  // scale, mode
            element::f16,
            ROIPoolingParams::increasinglyFilledBlob<float16>(1 * 8 * 8),
            std::vector<
                float16>{0.f, 0.15f, 0.2f, 0.75f, 0.8f, 0.f, 0.15f, 0.2f, 0.75f, 0.8f, 0.f, 0.15f, 0.2f, 0.75f, 0.8f},
            std::vector<float16>{1.225f,
                                 1.645f,
                                 4.585f,
                                 5.005f,
                                 1.225f,
                                 1.645f,
                                 4.585f,
                                 5.005f,
                                 1.225f,
                                 1.645f,
                                 4.585f,
                                 5.005f})),
    ReferenceRoiPoolingLayerTest::getTestCaseName);
