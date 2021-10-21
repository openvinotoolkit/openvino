// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <random>
#include "openvino/op/psroi_pooling.hpp"
#include "base_reference_test.hpp"
#include "openvino/opsets/opset1.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct DeformablePSROIPoolingParams {
    template <class IT>
    DeformablePSROIPoolingParams(const size_t batch_in, const size_t channel_in, const size_t height_in, const size_t width_in,
                                 const float spatial_scale, const size_t group_size, const int64_t spatial_bins_x, const int64_t spatial_bins_y,
                                 const float trans_std, const int64_t part_size, const size_t rois_dim,
                                 const ov::element::Type& iType, const bool is_input_generation_iota, const float inputValue, const float offsetValue,
                                 const std::vector<IT>& roisValues, const std::vector<IT>& oValues,
                                 const std::string& test_name = "", const std::string& mode = "bilinear_deformable")
        : groupSize(group_size),
          spatialBinsX(spatial_bins_x),
          spatialBinsY(spatial_bins_y),
          spatialScale(spatial_scale),
          transStd(trans_std),
          partSize(part_size),
          mode(mode),
          inputType(iType),
          roisType(iType),
          offsetsType(iType),
          outType(iType),
          roisData(CreateTensor(iType, roisValues)),
          refData(CreateTensor(iType, oValues)),
          testcaseName(test_name) {
              outputDim = channel_in / (group_size * group_size);
              inputShape = Shape{batch_in, channel_in, height_in, width_in};
              roisShape = Shape{rois_dim, 5};
              offsetsShape = Shape{rois_dim, 2, group_size, group_size};

              std::vector<IT> inputValues(shape_size(inputShape.get_shape()));
              if (is_input_generation_iota)
                  std::iota(inputValues.begin(), inputValues.end(), inputValue);
              else
                  std::fill(inputValues.begin(), inputValues.end(), inputValue);
              inputData = CreateTensor(iType, inputValues);

              std::vector<IT> offsetsValues(shape_size(offsetsShape.get_shape()));
              std::fill(offsetsValues.begin(), offsetsValues.end(), offsetValue);
              offsetsData = CreateTensor(iType, offsetsValues);
          }

    size_t groupSize;
    int64_t spatialBinsX;
    int64_t spatialBinsY;
    int64_t outputDim;
    float spatialScale;
    float transStd;
    int64_t partSize;

    std::string mode;
    ov::PartialShape inputShape;
    ov::PartialShape roisShape;
    ov::PartialShape offsetsShape;
    ov::element::Type inputType;
    ov::element::Type roisType;
    ov::element::Type offsetsType;
    ov::element::Type outType;
    ov::runtime::Tensor inputData;
    ov::runtime::Tensor roisData;
    ov::runtime::Tensor offsetsData;
    ov::runtime::Tensor refData;
    std::string testcaseName;
};

class ReferenceDeformablePSROIPoolingLayerTest : public testing::TestWithParam<DeformablePSROIPoolingParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        if (params.offsetsShape.size() != 0)
            inputData = {params.inputData, params.roisData, params.offsetsData};
        else
            inputData = {params.inputData, params.roisData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<DeformablePSROIPoolingParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "inputShape=" << param.inputShape << "_";
        result << "roiShape=" << param.roisShape << "_";
        if (param.offsetsShape.size() != 0)
            result << "offsetsShape=" << param.offsetsShape << "_";
        result << "iType=" << param.inputType << "_";
        if (param.testcaseName != "") {
            result << "mode=" << param.spatialScale << "_";
            result << param.testcaseName;
        } else {
            result << "mode=" << param.spatialScale;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const DeformablePSROIPoolingParams& params) {
        const auto input = std::make_shared<op::v0::Parameter>(params.inputType, params.inputShape);
        const auto rois = std::make_shared<op::v0::Parameter>(params.roisType, params.roisShape);
        if (params.offsetsShape.size() != 0) {
            const auto offsets = std::make_shared<op::v0::Parameter>(params.offsetsType, params.offsetsShape);
            const auto DeformablePSROIPooling = std::make_shared<opset1::DeformablePSROIPooling>(input,
                                                                            rois,
                                                                            offsets,
                                                                            params.outputDim,
                                                                            params.spatialScale,
                                                                            params.groupSize,
                                                                            params.mode,
                                                                            params.spatialBinsX,
                                                                            params.spatialBinsY,
                                                                            params.transStd,
                                                                            params.partSize);
            return std::make_shared<ov::Function>(NodeVector {DeformablePSROIPooling}, ParameterVector {input, rois, offsets});
        } else {
            const auto DeformablePSROIPooling = std::make_shared<opset1::DeformablePSROIPooling>(input,
                                                                            rois,
                                                                            params.outputDim,
                                                                            params.spatialScale,
                                                                            params.groupSize,
                                                                            params.mode,
                                                                            params.spatialBinsX,
                                                                            params.spatialBinsY,
                                                                            params.transStd,
                                                                            params.partSize);
            return std::make_shared<ov::Function>(NodeVector {DeformablePSROIPooling}, ParameterVector {input, rois});
        }
    }
};

TEST_P(ReferenceDeformablePSROIPoolingLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<DeformablePSROIPoolingParams> generateDeformablePSROIPoolingFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DeformablePSROIPoolingParams> deformablePSROIPoolingParams {
        DeformablePSROIPoolingParams(1, 16, 2, 2,
                                     0.0625, 2, 1, 1,
                                     1, 1, 2,
                                     IN_ET, true, 0, 0.0,
                                     std::vector<T>{
                                                    // input_batch_id, x1, y1, x2, y2
                                                    0,
                                                    1,
                                                    2,
                                                    4,
                                                    6,

                                                    0,
                                                    0,
                                                    3,
                                                    10,
                                                    4},
                                     std::vector<T>{
                                                    // First ROI
                                                    0,
                                                    4,
                                                    8,
                                                    12,

                                                    16,
                                                    20,
                                                    24,
                                                    28,

                                                    32,
                                                    36,
                                                    40,
                                                    44,

                                                    48,
                                                    52,
                                                    56,
                                                    60,

                                                    // Second ROI
                                                    0,
                                                    4,
                                                    8,
                                                    12,

                                                    16,
                                                    20,
                                                    24,
                                                    28,

                                                    32,
                                                    36,
                                                    40,
                                                    44,

                                                    48,
                                                    52,
                                                    56,
                                                    60}),
    };
    return deformablePSROIPoolingParams;
}

std::vector<DeformablePSROIPoolingParams> generateDeformablePSROIPoolingCombinedParams() {
    const std::vector<std::vector<DeformablePSROIPoolingParams>> deformablePSROIPoolingTypeParams {
        generateDeformablePSROIPoolingFloatParams<element::Type_t::f64>(),
        generateDeformablePSROIPoolingFloatParams<element::Type_t::f32>(),
        generateDeformablePSROIPoolingFloatParams<element::Type_t::f16>(),
        generateDeformablePSROIPoolingFloatParams<element::Type_t::bf16>()
        };
    std::vector<DeformablePSROIPoolingParams> combinedParams;

    for (const auto& params : deformablePSROIPoolingTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_DeformablePSROIPooling_With_Hardcoded_Refs, ReferenceDeformablePSROIPoolingLayerTest,
    testing::ValuesIn(generateDeformablePSROIPoolingCombinedParams()), ReferenceDeformablePSROIPoolingLayerTest::getTestCaseName);

} // namespace