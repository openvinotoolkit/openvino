// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/deformable_psroi_pooling.hpp"

#include <gtest/gtest.h>

#include <random>

#include "base_reference_test.hpp"
#include "openvino/op/psroi_pooling.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct DeformablePSROIPoolingParams {
    template <class IT>
    DeformablePSROIPoolingParams(const size_t batch_in,
                                 const size_t channel_in,
                                 const size_t height_in,
                                 const size_t width_in,
                                 const float spatial_scale,
                                 const size_t group_size,
                                 const int64_t spatial_bins_x,
                                 const int64_t spatial_bins_y,
                                 const float trans_std,
                                 const int64_t part_size,
                                 const size_t rois_dim,
                                 const ov::element::Type& iType,
                                 const bool is_input_generation_iota,
                                 const float inputValue,
                                 const std::vector<IT>& roisValues,
                                 const std::vector<IT>& oValues,
                                 const std::string& test_name = "",
                                 const std::string& mode = "bilinear_deformable")
        : groupSize(group_size),
          spatialBinsX(spatial_bins_x),
          spatialBinsY(spatial_bins_y),
          spatialScale(spatial_scale),
          transStd(trans_std),
          partSize(part_size),
          mode(mode),
          inputType(iType),
          roisType(iType),
          outType(iType),
          testcaseName(test_name) {
        outputDim = (channel_in / (group_size * group_size)) -
                    (static_cast<size_t>(channel_in / (group_size * group_size)) % 2);
        inputShape = Shape{batch_in, channel_in, height_in, width_in};

        roisShape = Shape{rois_dim, 5};
        roisData = CreateTensor(roisShape, iType, roisValues);

        std::vector<IT> inputValues(shape_size(inputShape));
        if (is_input_generation_iota)
            std::iota(inputValues.begin(), inputValues.end(), inputValue);
        else
            std::fill(inputValues.begin(), inputValues.end(), inputValue);
        inputData = CreateTensor(inputShape, iType, inputValues);

        const auto output_shape = Shape{rois_dim, outputDim, group_size, group_size};
        if (oValues.size() > 1) {
            refData = CreateTensor(output_shape, iType, oValues);
        } else {
            std::vector<IT> expected_output_values(shape_size(output_shape));
            std::fill(expected_output_values.begin(), expected_output_values.end(), oValues[0]);
            refData = CreateTensor(output_shape, iType, expected_output_values);
        }
    }

    template <class IT>
    DeformablePSROIPoolingParams(const size_t batch_in,
                                 const size_t channel_in,
                                 const size_t height_in,
                                 const size_t width_in,
                                 const float spatial_scale,
                                 const size_t group_size,
                                 const int64_t spatial_bins_x,
                                 const int64_t spatial_bins_y,
                                 const float trans_std,
                                 const int64_t part_size,
                                 const size_t rois_dim,
                                 const ov::element::Type& iType,
                                 const bool is_input_generation_iota,
                                 const float inputValue,
                                 const float offsetValue,
                                 const std::vector<IT>& roisValues,
                                 const std::vector<IT>& oValues,
                                 const std::string& test_name = "",
                                 const std::string& mode = "bilinear_deformable")
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
          testcaseName(test_name) {
        outputDim = (channel_in / (group_size * group_size)) - ((channel_in / (group_size * group_size)) % 2);
        inputShape = Shape{batch_in, channel_in, height_in, width_in};
        offsetsShape = Shape{rois_dim, 2, group_size, group_size};

        roisShape = Shape{rois_dim, 5};
        roisData = CreateTensor(roisShape, iType, roisValues);

        std::vector<IT> inputValues(shape_size(inputShape));
        if (is_input_generation_iota)
            std::iota(inputValues.begin(), inputValues.end(), inputValue);
        else
            std::fill(inputValues.begin(), inputValues.end(), inputValue);
        inputData = CreateTensor(inputShape, iType, inputValues);

        std::vector<IT> offsetsValues(shape_size(offsetsShape));
        std::fill(offsetsValues.begin(), offsetsValues.end(), offsetValue);
        offsetsData = CreateTensor(offsetsShape, iType, offsetsValues);

        const auto output_shape = Shape{rois_dim, outputDim, group_size, group_size};
        if (oValues.size() > 1) {
            refData = CreateTensor(output_shape, iType, oValues);
        } else {
            std::vector<IT> expected_output_values(shape_size(output_shape));
            std::fill(expected_output_values.begin(), expected_output_values.end(), oValues[0]);
            refData = CreateTensor(output_shape, iType, expected_output_values);
        }
    }

    size_t groupSize;
    int64_t spatialBinsX;
    int64_t spatialBinsY;
    size_t outputDim;
    float spatialScale;
    float transStd;
    int64_t partSize;

    std::string mode;
    ov::Shape inputShape;
    ov::Shape roisShape;
    ov::Shape offsetsShape;
    ov::element::Type inputType;
    ov::element::Type roisType;
    ov::element::Type offsetsType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor roisData;
    ov::Tensor offsetsData;
    ov::Tensor refData;
    std::string testcaseName;
};

class ReferenceDeformablePSROIPoolingLayerTest : public testing::TestWithParam<DeformablePSROIPoolingParams>,
                                                 public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        if (params.offsetsShape.size() != 0)
            inputData = {params.inputData, params.roisData, params.offsetsData};
        else
            inputData = {params.inputData, params.roisData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<DeformablePSROIPoolingParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "inputShape=" << param.inputShape << "_";
        result << "roiShape=" << param.roisShape << "_";
        if (param.offsetsShape.size() != 0)
            result << "offsetsShape=" << param.offsetsShape << "_";
        result << "outputDim=" << param.outputDim << "_";
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
    static std::shared_ptr<Model> CreateFunction(const DeformablePSROIPoolingParams& params) {
        const auto input = std::make_shared<op::v0::Parameter>(params.inputType, params.inputShape);
        const auto rois = std::make_shared<op::v0::Parameter>(params.roisType, params.roisShape);
        if (params.offsetsShape.size() != 0) {
            const auto offsets = std::make_shared<op::v0::Parameter>(params.offsetsType, params.offsetsShape);
            const auto DeformablePSROIPooling = std::make_shared<op::v1::DeformablePSROIPooling>(input,
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
            return std::make_shared<ov::Model>(NodeVector{DeformablePSROIPooling},
                                               ParameterVector{input, rois, offsets});
        } else {
            const auto DeformablePSROIPooling = std::make_shared<op::v1::DeformablePSROIPooling>(input,
                                                                                                 rois,
                                                                                                 params.outputDim,
                                                                                                 params.spatialScale,
                                                                                                 params.groupSize,
                                                                                                 params.mode,
                                                                                                 params.spatialBinsX,
                                                                                                 params.spatialBinsY,
                                                                                                 params.transStd,
                                                                                                 params.partSize);
            return std::make_shared<ov::Model>(NodeVector{DeformablePSROIPooling}, ParameterVector{input, rois});
        }
    }
};

TEST_P(ReferenceDeformablePSROIPoolingLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<DeformablePSROIPoolingParams> generateDeformablePSROIPoolingFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DeformablePSROIPoolingParams> deformablePSROIPoolingParams{
        DeformablePSROIPoolingParams(1,
                                     16,
                                     2,
                                     2,  // batch_in, channel_in, height_in, width_in
                                     0.0625,
                                     2,
                                     1,
                                     1,  // spatial_scale, group_size, spatial_bins_x, spatial_bins_y
                                     1,
                                     1,
                                     2,  // trans_std, part_size, rois_dim
                                     IN_ET,
                                     true,
                                     0,
                                     0.0,           // inputType, is_input_generation_iota, inputValue, offsetValue
                                     std::vector<T>{// input_batch_id, x1, y1, x2, y2
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
                                     std::vector<T>{// First ROI
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
                                                    60},
                                     "offset_00"),
        DeformablePSROIPoolingParams(1,
                                     16,
                                     2,
                                     2,  // batch_in, channel_in, height_in, width_in
                                     0.0625,
                                     2,
                                     1,
                                     1,  // spatial_scale, group_size, spatial_bins_x, spatial_bins_y
                                     1,
                                     1,
                                     2,  // trans_std, part_size, rois_dim
                                     IN_ET,
                                     true,
                                     0,
                                     0.2,           // inputType, is_input_generation_iota, inputValue, offsetValue
                                     std::vector<T>{// input_batch_id, x1, y1, x2, y2
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
                                     std::vector<T>{// First ROI
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
                                                    60},
                                     "offset_0p2"),
        DeformablePSROIPoolingParams(1,
                                     16,
                                     2,
                                     2,  // batch_in, channel_in, height_in, width_in
                                     0.0625,
                                     2,
                                     1,
                                     1,  // spatial_scale, group_size, spatial_bins_x, spatial_bins_y
                                     1,
                                     1,
                                     2,  // trans_std, part_size, rois_dim
                                     IN_ET,
                                     true,
                                     0,
                                     0.5,           // inputType, is_input_generation_iota, inputValue, offsetValue
                                     std::vector<T>{// input_batch_id, x1, y1, x2, y2
                                                    0,
                                                    1,
                                                    2,
                                                    4,
                                                    6,
                                                    0,
                                                    5,
                                                    3,
                                                    10,
                                                    4},
                                     std::vector<T>{// First ROI
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
                                                    4.1875,
                                                    8,
                                                    12.1875,
                                                    16,
                                                    20.1875,
                                                    24,
                                                    28.1875,
                                                    32,
                                                    36.1875,
                                                    40,
                                                    44.1875,
                                                    48,
                                                    52.1875,
                                                    56,
                                                    60.1875},
                                     "offset_0p5"),
        DeformablePSROIPoolingParams(1,
                                     16,
                                     2,
                                     2,  // batch_in, channel_in, height_in, width_in
                                     0.0625,
                                     2,
                                     1,
                                     1,  // spatial_scale, group_size, spatial_bins_x, spatial_bins_y
                                     1,
                                     1,
                                     2,  // trans_std, part_size, rois_dim
                                     IN_ET,
                                     true,
                                     0,
                                     0,             // inputType, is_input_generation_iota, inputValue, offsetValue
                                     std::vector<T>{// input_batch_id, x1, y1, x2, y2
                                                    0,
                                                    10,
                                                    10,
                                                    20,
                                                    20,
                                                    0,
                                                    100,
                                                    100,
                                                    200,
                                                    200},
                                     std::vector<T>{// First ROI
                                                    0.375,
                                                    4.71875,
                                                    9.0625,
                                                    13.40625,
                                                    16.375,
                                                    20.71875,
                                                    25.0625,
                                                    29.40625,
                                                    32.375,
                                                    36.71875,
                                                    41.0625,
                                                    45.40625,
                                                    48.375,
                                                    52.71875,
                                                    57.0625,
                                                    61.40625,
                                                    // Second ROI
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0},
                                     "roi_oversize"),
        DeformablePSROIPoolingParams(1,
                                     8,
                                     3,
                                     3,  // batch_in, channel_in, height_in, width_in
                                     1,
                                     2,
                                     1,
                                     1,  // spatial_scale, group_size, spatial_bins_x, spatial_bins_y
                                     1,
                                     2,
                                     1,  // trans_std, part_size, rois_dim
                                     IN_ET,
                                     true,
                                     0,             // inputType, is_input_generation_iota, inputValue, offsetValue
                                     std::vector<T>{// input_batch_id, x1, y1, x2, y2
                                                    0,
                                                    1,
                                                    1,
                                                    2,
                                                    2},
                                     std::vector<T>{2.0, 12.0, 23.0, 33.0, 38.0, 48.0, 59.0, 69.0},
                                     "no_offset_input"),
        DeformablePSROIPoolingParams(1,
                                     8,
                                     3,
                                     3,  // batch_in, channel_in, height_in, width_in
                                     1,
                                     2,
                                     1,
                                     1,  // spatial_scale, group_size, spatial_bins_x, spatial_bins_y
                                     1,
                                     2,
                                     1,  // trans_std, part_size, rois_dim,
                                     IN_ET,
                                     true,
                                     0,
                                     0,             // inputType, is_input_generation_iota, inputValue, offsetValue
                                     std::vector<T>{// input_batch_id, x1, y1, x2, y2
                                                    0,
                                                    1,
                                                    1,
                                                    2,
                                                    2},
                                     std::vector<T>{2.0, 12.0, 23.0, 33.0, 38.0, 48.0, 59.0, 69.0},
                                     "offset_zero"),
        DeformablePSROIPoolingParams(1,
                                     8,
                                     3,
                                     3,  // batch_in, channel_in, height_in, width_in
                                     1,
                                     2,
                                     1,
                                     1,  // spatial_scale, group_size, spatial_bins_x, spatial_bins_y
                                     1,
                                     2,
                                     1,  // trans_std, part_size, rois_dim,
                                     IN_ET,
                                     true,
                                     0,
                                     0.1,           // inputType, is_input_generation_iota, inputValue, offsetValue
                                     std::vector<T>{// input_batch_id, x1, y1, x2, y2
                                                    0,
                                                    1,
                                                    1,
                                                    2,
                                                    2},
                                     std::vector<T>{2.8, 12.8, 23.8, 33.8, 38.8, 48.8, 59.8, 69.8},
                                     "offset_01"),
        DeformablePSROIPoolingParams(1,
                                     8,
                                     3,
                                     3,  // batch_in, channel_in, height_in, width_in
                                     1,
                                     2,
                                     1,
                                     1,  // spatial_scale, group_size, spatial_bins_x, spatial_bins_y
                                     1,
                                     2,
                                     1,  // trans_std, part_size, rois_dim,
                                     IN_ET,
                                     true,
                                     0,
                                     0.5,           // inputType, is_input_generation_iota, inputValue, offsetValue
                                     std::vector<T>{// input_batch_id, x1, y1, x2, y2
                                                    0,
                                                    1,
                                                    1,
                                                    2,
                                                    2},
                                     std::vector<T>{6., 15.5, 25.5, 35., 42., 51.5, 61.5, 71.},
                                     "offset_05"),
        DeformablePSROIPoolingParams(
            1,
            16,
            2,
            2,  // batch_in, channel_in, height_in, width_in
            0.0625,
            2,
            1,
            1,  // spatial_scale, group_size, spatial_bins_x, spatial_bins_y
            1,
            1,
            1,  // trans_std, part_size, rois_dim,
            IN_ET,
            false,
            0.1,
            0.1,           // inputType, is_input_generation_iota, inputValue, offsetValue
            std::vector<T>{// input_batch_id, x1, y1, x2, y2
                           0,
                           10,
                           10,
                           10,
                           10},
            std::vector<T>{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
            "single_value"),
        DeformablePSROIPoolingParams(1,
                                     1024,
                                     63,
                                     38,  // batch_in, channel_in, height_in, width_in
                                     0.0625,
                                     3,
                                     1,
                                     1,  // spatial_scale, group_size, spatial_bins_x, spatial_bins_y
                                     1,
                                     1,
                                     2,  // trans_std, part_size, rois_dim,
                                     IN_ET,
                                     false,
                                     0.1,
                                     0.0,           // inputType, is_input_generation_iota, inputValue, offsetValue
                                     std::vector<T>{// input_batch_id, x1, y1, x2, y2
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
                                     std::vector<T>{0.1},
                                     "single_value_big_shape")};
    return deformablePSROIPoolingParams;
}

std::vector<DeformablePSROIPoolingParams> generateDeformablePSROIPoolingCombinedParams() {
    const std::vector<std::vector<DeformablePSROIPoolingParams>> deformablePSROIPoolingTypeParams{
        generateDeformablePSROIPoolingFloatParams<element::Type_t::f64>(),
        generateDeformablePSROIPoolingFloatParams<element::Type_t::f32>(),
        generateDeformablePSROIPoolingFloatParams<element::Type_t::f16>(),
        generateDeformablePSROIPoolingFloatParams<element::Type_t::bf16>()};
    std::vector<DeformablePSROIPoolingParams> combinedParams;

    for (const auto& params : deformablePSROIPoolingTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_DeformablePSROIPooling_With_Hardcoded_Refs,
                         ReferenceDeformablePSROIPoolingLayerTest,
                         testing::ValuesIn(generateDeformablePSROIPoolingCombinedParams()),
                         ReferenceDeformablePSROIPoolingLayerTest::getTestCaseName);

}  // namespace
