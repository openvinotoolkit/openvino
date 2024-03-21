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
    template <class IT>
    ROIAlignRotatedParams(const PartialShape& pShape,
                   const element::Type& iType,
                   const std::vector<IT>& iValues,
                   const reference_tests::Tensor& expectedFeatureMap,
                   const reference_tests::Tensor& coords,
                   const reference_tests::Tensor& roiIdx,
                   const int32_t pooledH,
                   const int32_t pooledW,
                   const float spatialScale,
                   const int32_t poolingRatio,
                   const bool clockwise,
                   const std::string& testcaseName)
        : pShape(pShape),
          iType(iType),
          featureMap(CreateTensor(iType, iValues)),
          expectedFeatureMap(expectedFeatureMap),
          coords(coords),
          roiIdx(roiIdx),
          pooledH(pooledH),
          pooledW(pooledW),
          spatialScale(spatialScale),
          poolingRatio(poolingRatio),
          clockwise(clockwise),
          testcaseName(testcaseName) {}

    PartialShape pShape;
    element::Type iType;
    ov::Tensor featureMap;
    reference_tests::Tensor expectedFeatureMap;
    reference_tests::Tensor coords;
    reference_tests::Tensor roiIdx;
    int32_t pooledH;
    int32_t pooledW;
    float spatialScale;
    int32_t poolingRatio;
    bool clockwise;
    std::string testcaseName;
};

class ReferenceROIAlignRotatedTest : public testing::TestWithParam<ROIAlignRotatedParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.featureMap};
        refOutData = {params.expectedFeatureMap.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ROIAlignRotatedParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.iType;
        result << "_pShape=" << param.pShape;
        result << "_efType=" << param.expectedFeatureMap.type;
        result << "_efShape=" << param.expectedFeatureMap.shape;
        result << "_cType=" << param.coords.type;
        result << "_cShape=" << param.coords.shape;
        result << "_rType=" << param.roiIdx.type;
        result << "_rShape=" << param.roiIdx.shape;
        result << "_pooledH=" << param.pooledH;
        result << "_pooledW=" << param.pooledW;
        result << "_spatialScale=" << param.spatialScale;
        result << "_clockwise=" << param.clockwise;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ROIAlignRotatedParams& params) {
        const auto featureMap = std::make_shared<op::v0::Parameter>(params.iType, params.pShape);
        const auto coords =
            std::make_shared<op::v0::Constant>(params.coords.type, params.coords.shape, params.coords.data.data());
        const auto roisIdx =
            std::make_shared<op::v0::Constant>(params.roiIdx.type, params.roiIdx.shape, params.roiIdx.data.data());
        const auto roi_align_rot = std::make_shared<op::v14::ROIAlignRotated>(featureMap,
                                                                  coords,
                                                                  roisIdx,
                                                                  params.pooledH,
                                                                  params.pooledW,
                                                                  params.poolingRatio,
                                                                  params.spatialScale,
                                                                  params.clockwise);
        auto f = std::make_shared<Model>(NodeVector{roi_align_rot}, ParameterVector{featureMap});
        return f;
    }
};

TEST_P(ReferenceROIAlignRotatedTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ROIAlignRotatedParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ROIAlignRotatedParams> params{
        ROIAlignRotatedParams(
            PartialShape{2, 1, 8, 8},
            ET,
            std::vector<T>{0, 1, 8, 5, 5,  2, 0,  7, 7, 10, 4,  5, 9,  0, 0, 5,  7,  0, 4, 0, 4, 7,  6,  10, 9,  5,
                           1, 7, 4, 7, 10, 8, 2,  0, 8, 3,  6,  8, 10, 4, 2, 10, 7,  8, 7, 0, 6, 9,  2,  4,  8,  5,
                           2, 3, 3, 1, 5,  9, 10, 0, 9, 5,  5,  3, 10, 5, 2, 0,  10, 0, 5, 4, 3, 10, 5,  5,  10, 0,
                           8, 8, 9, 1, 0,  7, 9,  6, 8, 7,  10, 9, 2,  3, 3, 5,  6,  9, 4, 9, 2, 4,  5,  5,  3,  1,
                           1, 6, 8, 0, 5,  5, 10, 8, 6, 9,  6,  9, 1,  2, 7, 1,  1,  3, 0, 4, 0, 7,  10, 2},
            reference_tests::Tensor(ET, {2, 1, 2, 2}, std::vector<T>{3, 3.75, 4.75, 5, 3, 5.5, 2.75, 3.75}),
            reference_tests::Tensor(ET, {2, 5}, std::vector<T>{3.5, 3.5, 2, 2, 0, 3.5, 3.5, 2, 2, 0}),
            reference_tests::Tensor(element::Type_t::i32, {2}, std::vector<int32_t>{0, 1}),
            2,
            2,
            1,
            2,
            true,
            "roi_align_rotated_angle_0"),
        ROIAlignRotatedParams(
            PartialShape{1, 1, 2, 2},
            ET,
            std::vector<T>{1,2,3,4},
            reference_tests::Tensor(ET, {1, 1, 2, 2}, std::vector<T>{1.0, 1.25, 1.5, 1.75}),
            reference_tests::Tensor(ET, {1, 5}, std::vector<T>{0.5, 0.5, 1, 1, 0}),
            reference_tests::Tensor(element::Type_t::i32, {1}, std::vector<int32_t>{0}),
            2,
            2,
            1,
            2,
            true,
            "roi_align_rotated_simple_angle_0"),
        ROIAlignRotatedParams(
            PartialShape{1, 1, 2, 2},
            ET,
            std::vector<T>{1,2,3,4},
            reference_tests::Tensor(ET, {1, 1, 2, 2}, std::vector<T>{1.5, 1.0, 1.75, 1.25}),
            reference_tests::Tensor(ET, {1, 5}, std::vector<T>{0.5, 0.5, 1, 1, PI/2}),
            reference_tests::Tensor(element::Type_t::i32, {1}, std::vector<int32_t>{0}),
            2,
            2,
            1,
            2,
            false,
            "roi_align_rotated_simple_angle_PI/2"),
        ROIAlignRotatedParams(
            PartialShape{1, 1, 2, 2},
            ET,
            std::vector<T>{1,2,3,4},
            reference_tests::Tensor(ET, {2, 1, 2, 2}, std::vector<T>{1.75, 1.5, 1.25, 1.0, 1.0, 1.25, 1.5, 1.75}),
            reference_tests::Tensor(ET, {2, 5}, std::vector<T>{0.5, 0.5, 1, 1, PI, 0.5, 0.5, 1, 1, 2*PI}),
            reference_tests::Tensor(element::Type_t::i32, {2}, std::vector<int32_t>{0, 0}),
            2,
            2,
            1,
            2,
            false,
            "roi_align_rotated_batch_idx_test"),
    };
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
