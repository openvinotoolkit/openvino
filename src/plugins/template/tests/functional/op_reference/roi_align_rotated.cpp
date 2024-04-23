// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_align_rotated.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

constexpr double PI = 3.141592653589793238462643383279;

struct ROIAlignRotatedParams {
    PartialShape inputShape;
    int32_t pooledH;
    int32_t pooledW;
    float spatialScale;
    int32_t samplingRatio;
    bool clockwise;
    std::string testcaseName;
    ov::Tensor input;
    reference_tests::Tensor rois;
    reference_tests::Tensor roiBatchIdxs;
    reference_tests::Tensor expectedOutput;
};

template <typename T>
ROIAlignRotatedParams PrepareTestCaseParams(const PartialShape& inputShape,
                                            size_t pooledH,
                                            size_t pooledW,
                                            float spatialScale,
                                            int32_t samplingRatio,
                                            bool clockwise,
                                            const std::vector<T>& inputValues,
                                            const std::vector<T>& roisVals,
                                            const std::vector<int32_t>& roiBatchIdx,
                                            const std::vector<T>& expectedValues,
                                            const std::string& testcaseName) {
    ROIAlignRotatedParams ret;

    constexpr size_t rois_second_dim_size = 5;  //< By definition of the ROIAlignRotated op

    const auto numOfRois = roisVals.size() / rois_second_dim_size;
    const auto channels = static_cast<size_t>(inputShape[1].get_length());
    const auto elementType = element::from<T>();

    ret.inputShape = inputShape;
    ret.pooledH = pooledH;
    ret.pooledW = pooledW;
    ret.spatialScale = spatialScale;
    ret.samplingRatio = samplingRatio;
    ret.clockwise = clockwise;
    ret.testcaseName = testcaseName;
    ret.input = CreateTensor(elementType, inputValues);
    ret.rois = reference_tests::Tensor(elementType, {numOfRois, 5}, roisVals);
    ret.roiBatchIdxs = reference_tests::Tensor(element::Type_t::i32, {numOfRois}, roiBatchIdx);
    ret.expectedOutput = reference_tests::Tensor(elementType, {numOfRois, channels, pooledH, pooledW}, expectedValues);

    return ret;
}

class ReferenceROIAlignRotatedTest : public testing::TestWithParam<ROIAlignRotatedParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.input};
        refOutData = {params.expectedOutput.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ROIAlignRotatedParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "type=" << param.input.get_element_type();
        result << "_inputShape=" << param.inputShape;
        result << "_pooledH=" << param.pooledH;
        result << "_pooledW=" << param.pooledW;
        result << "_spatialScale=" << param.spatialScale;
        result << "_samplingRatio=" << param.samplingRatio;
        result << "_clockwise=" << param.clockwise;
        result << "_roisShape=" << param.rois.shape;
        result << "_roisBatchIdxShape=" << param.roiBatchIdxs.shape;
        result << "_outputShape=" << param.expectedOutput.shape;
        if (!param.testcaseName.empty()) {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ROIAlignRotatedParams& params) {
        const auto featureMap = std::make_shared<op::v0::Parameter>(params.input.get_element_type(), params.inputShape);
        const auto coords =
            std::make_shared<op::v0::Constant>(params.rois.type, params.rois.shape, params.rois.data.data());
        const auto roisIdx = std::make_shared<op::v0::Constant>(params.roiBatchIdxs.type,
                                                                params.roiBatchIdxs.shape,
                                                                params.roiBatchIdxs.data.data());
        const auto roi_align_rot = std::make_shared<op::v14::ROIAlignRotated>(featureMap,
                                                                              coords,
                                                                              roisIdx,
                                                                              params.pooledH,
                                                                              params.pooledW,
                                                                              params.samplingRatio,
                                                                              params.spatialScale,
                                                                              params.clockwise);
        return std::make_shared<Model>(NodeVector{roi_align_rot}, ParameterVector{featureMap});
    }
};

TEST_P(ReferenceROIAlignRotatedTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ROIAlignRotatedParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ROIAlignRotatedParams> params;
    // NOTE: expected output were generated using mmvc roi_align_rotated implementation.
    params.push_back(PrepareTestCaseParams<T>(
        {2, 1, 8, 8},
        2,
        2,
        1.0f,
        2,
        true,
        {0,  1, 8, 5, 5,  2, 0,  7, 7, 10, 4, 5, 9,  0, 0,  5, 7, 0, 4, 0, 4, 7, 6, 10, 9,  5, 1,  7, 4, 7, 10, 8,
         2,  0, 8, 3, 6,  8, 10, 4, 2, 10, 7, 8, 7,  0, 6,  9, 2, 4, 8, 5, 2, 3, 3, 1,  5,  9, 10, 0, 9, 5, 5,  3,
         10, 5, 2, 0, 10, 0, 5,  4, 3, 10, 5, 5, 10, 0, 8,  8, 9, 1, 0, 7, 9, 6, 8, 7,  10, 9, 2,  3, 3, 5, 6,  9,
         4,  9, 2, 4, 5,  5, 3,  1, 1, 6,  8, 0, 5,  5, 10, 8, 6, 9, 6, 9, 1, 2, 7, 1,  1,  3, 0,  4, 0, 7, 10, 2},
        {3.5, 3.5, 2, 2, 0, 3.5, 3.5, 2, 2, 0},
        {0, 1},
        {3, 3.75, 4.75, 5, 3, 5.5, 2.75, 3.75},
        "roi_align_rotated_angle_0"));
    params.push_back(PrepareTestCaseParams<T>({1, 1, 2, 2},
                                              2,
                                              2,
                                              1.0f,
                                              2,
                                              true,
                                              {1, 2, 3, 4},
                                              {0.5, 0.5, 1, 1, 0},
                                              {0},
                                              {1.0, 1.25, 1.5, 1.75},
                                              "roi_align_rotated_simple_angle_0"));
    params.push_back(PrepareTestCaseParams<T>({1, 1, 2, 2},
                                              2,
                                              2,
                                              1.0f,
                                              2,
                                              false,
                                              {1, 2, 3, 4},
                                              {0.5, 0.5, 1, 1, PI / 2},
                                              {0},
                                              {1.5, 1.0, 1.75, 1.25},
                                              "roi_align_rotated_simple_angle_PI/2"));
    params.push_back(PrepareTestCaseParams<T>({1, 1, 2, 2},
                                              2,
                                              2,
                                              1.0f,
                                              2,
                                              false,
                                              {1, 2, 3, 4},
                                              {0.5, 0.5, 1, 1, PI, 0.5, 0.5, 1, 1, 2 * PI},
                                              {0, 0},
                                              {1.75, 1.5, 1.25, 1.0, 1.0, 1.25, 1.5, 1.75},
                                              "roi_align_rotated_batch_idx_test"));
    params.push_back(PrepareTestCaseParams<T>({1, 2, 2, 2},
                                              2,
                                              2,
                                              1.0f,
                                              2,
                                              false,
                                              {1, 2, 3, 4, 4, 3, 2, 1},
                                              {0.5, 0.5, 1, 1, 0},
                                              {0},
                                              {1.0, 1.25, 1.5, 1.75, 4.0, 3.75, 3.5, 3.25},
                                              "roi_align_rotated_channels_test"));
    params.push_back(PrepareTestCaseParams<T>(
        {1, 1, 5, 5},
        3,
        1,
        1.0f,
        2,
        true,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
        {1, 1, 4, 4, 0},
        {0},
        {0.8750, 4.2500, 10.9167},
        "roi_align_rotated_box_outside_feature_map_top_left"));
    params.push_back(PrepareTestCaseParams<T>(
        {1, 1, 5, 5},
        3,
        1,
        1.0f,
        2,
        true,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
        {1, 1, 4, 4, PI / 4},
        {0},
        {2.6107, 4.6642, 6.8819},
        "roi_align_rotated_box_outside_feature_map_top_left_angle_PI/4"));
    params.push_back(PrepareTestCaseParams<T>(
        {1, 1, 5, 5},
        3,
        1,
        1.0f,
        2,
        true,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
        {5, 5, 4, 4, 0},
        {0},
        {10.1667, 12.2500, 0.0},
        "roi_align_rotated_box_outside_feature_map_bottom_right_angle_0"));
    params.push_back(PrepareTestCaseParams<T>(
        {1, 1, 5, 5},
        3,
        1,
        1.0f,
        2,
        true,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
        {5, 5, 1, 5, PI / 4},
        {0},
        {0.0, 25.0, 0.0},
        "roi_align_rotated_box_outside_feature_map_bottom_right_angle_PI/4"));
    params.push_back(PrepareTestCaseParams<T>(
        {1, 1, 5, 5},
        2,
        2,
        1.0f,
        0,
        true,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
        {3, 3, 4, 4, 0},
        {0},
        {10.0, 12.0, 20.0, 22.0},
        "roi_align_rotated_box_outside_sampling_ratio_auto"));
    params.push_back(PrepareTestCaseParams<T>(
        {1, 1, 5, 5},
        2,
        2,
        0.25f,
        0,
        true,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
        {3, 3, 4, 4, 0},
        {0},
        {1.0, 1.5, 3.5, 4.0},
        "roi_align_rotated_box_outside_sampling_ratio_auto_scale_0.25"));
    params.push_back(PrepareTestCaseParams<T>(
        {1, 1, 5, 5},
        2,
        2,
        2.0f,
        0,
        true,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
        {3, 3, 4, 4, 0},
        {0},
        {20.5, 0.0, 0.0, 0.0},
        "roi_align_rotated_box_outside_sampling_ratio_auto_scale_2"));
    params.push_back(PrepareTestCaseParams<T>(
        {1, 1, 5, 5},
        5,
        2,
        0.78f,
        0,
        false,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
        {3, 1, 4, 2, PI / 3},
        {0},
        {5.1271, 1.2473, 6.1773, 2.9598, 7.2275, 3.2300, 8.2777, 3.7458, 9.3279, 4.4060},
        "roi_align_rotated_all_features"));
    return params;
}

std::vector<ROIAlignRotatedParams> generateCombinedParams() {
    const std::vector<std::vector<ROIAlignRotatedParams>> generatedParams{
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<ROIAlignRotatedParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ROIAlignRotated_With_Hardcoded_Refs,
                         ReferenceROIAlignRotatedTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceROIAlignRotatedTest::getTestCaseName);
}  // namespace
